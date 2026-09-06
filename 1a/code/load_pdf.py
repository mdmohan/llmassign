import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import pdfplumber


# --- Worker functions placed at module-level for clean multiprocessing pickling ---

def _calculate_md5(file_path: Path) -> str:
    """Calculate the MD5 hash of a file by reading it in binary chunks."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_existing_md5(txt_path: Path) -> str | None:
    """Read the first line of an existing text file to extract the stored MD5 hash."""
    if not txt_path.exists():
        return None
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line.startswith("# MD5:"):
                return first_line.split("# MD5:")[1].strip()
    except Exception:
        return None
    return None


def _is_char_in_bbox(char: dict, bbox: tuple) -> bool:
    """Check if a character's coordinates fall inside a table bounding box."""
    c_x0 = char.get("x0", 0)
    c_top = char.get("top", 0)
    c_x1 = char.get("x1", 0)
    c_bottom = char.get("bottom", 0)

    t_x0, t_top, t_x1, t_bottom = bbox
    return not (c_x1 < t_x0 or c_x0 > t_x1 or c_bottom < t_top or c_top > t_bottom)


def _format_table_as_markdown(table: list) -> str:
    """Convert a 2D list (table) into a Markdown grid string."""
    if not table or not any(table):
        return ""

    clean_table = [
        [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
        for row in table
    ]
    cols = max(len(row) for row in clean_table)
    clean_table = [row + [""] * (cols - len(row)) for row in clean_table]

    headers = clean_table[0]
    rows = clean_table[1:]

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * cols) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)


def strip_metadata_headers(text: str) -> str:
    """Removes lines starting with '# MD5:' or similar comment markers."""
    cleaned_text = re.sub(r"^#\s*MD5:\s*[a-fA-F0-9]{32}\s*\n?", "", text, flags=re.MULTILINE)
    return cleaned_text.strip()


def _process_single_pdf_worker(args: tuple) -> tuple[str, dict]:
    pdf_path, output_filepath, x_tolerance, y_tolerance, extract_tables = args
    pdf_path = Path(pdf_path)
    output_filepath = Path(output_filepath)

    current_md5 = _calculate_md5(pdf_path)

    # Check if output exists and matches hash
    existing_md5 = _read_existing_md5(output_filepath)
    if existing_md5 == current_md5:
        print(f"[SKIP] Unmodified: {pdf_path.name}", flush=True)
        
        # Read disk content and clean header via regex function
        raw_disk_text = output_filepath.read_text(encoding="utf-8")
        clean_text = strip_metadata_headers(raw_disk_text)
        
        return str(pdf_path), {
            "extracted_text": clean_text,
            "output_filename": str(output_filepath),
            "md5": current_md5,
            "status": "skipped_unmodified",
        }

    # Process new or modified file
    print(
        f"[PROCESS] ({'modified' if existing_md5 else 'new'}): {pdf_path.name}",
        flush=True,
    )
    page_text_blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                if extract_tables:
                    # Detect tables only once. page.extract_tables() performs its
                    # own table search, which duplicates expensive layout work.
                    tables = page.find_tables()
                    table_bboxes = [table.bbox for table in tables]
                    extracted_tables = [table.extract() for table in tables]
                else:
                    table_bboxes = []
                    extracted_tables = []

                def filter_out_tables(obj):
                    if obj.get("object_type") == "char":
                        return not any(
                            _is_char_in_bbox(obj, bbox) for bbox in table_bboxes
                        )
                    return True

                filtered_page = page.filter(filter_out_tables)
                raw_text = (
                    filtered_page.extract_text(
                        x_tolerance=x_tolerance, y_tolerance=y_tolerance
                    )
                    or ""
                )
                formatted_md_tables = [
                    formatted_table
                    for table in extracted_tables
                    if (formatted_table := _format_table_as_markdown(table))
                ]

                page_content = []
                if raw_text.strip():
                    page_content.append(raw_text.strip())

                if formatted_md_tables:
                    page_content.append("\n[EXTRACTED TABLES]:\n")
                    page_content.extend(formatted_md_tables)

                if page_content:
                    page_text_blocks.append("\n\n".join(page_content))
            finally:
                # pdfplumber caches page layout and character objects. Release
                # them immediately instead of retaining them for the whole PDF.
                close_page = getattr(page, "close", None)
                if callable(close_page):
                    close_page()
                else:
                    flush_cache = getattr(page, "flush_cache", None)
                    if callable(flush_cache):
                        flush_cache()

    clean_extracted_text = "\n\n".join(page_text_blocks).strip()

    # Save to disk with MD5 header for caching
    disk_file_content = f"# MD5: {current_md5}\n\n{clean_extracted_text}"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_filepath.write_text(disk_file_content, encoding="utf-8")

    return str(pdf_path), {
        "extracted_text": clean_extracted_text,
        "output_filename": str(output_filepath),
        "md5": current_md5,
        "status": "processed",
    }


class PdfToTxt:
    """
    Recursively loads mixed PDF and text corpora.

    PDFs are extracted with multiprocessing, while existing text files are read
    directly. PDF tables can be preserved as Markdown.
    """

    def __init__(
        self,
        x_tolerance: float = 2.0,
        y_tolerance: float = 3.0,
        max_workers: int | None = None,
        extract_tables: bool = True,
        max_tasks_per_child: int = 2,
    ):
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance
        available_cpus = os.cpu_count() or 1
        # PDF table extraction can consume substantial memory. Two workers are
        # a safer notebook default than using every available CPU.
        requested_workers = 2 if max_workers is None else max_workers
        self.max_workers = max(1, min(requested_workers, available_cpus))
        self.extract_tables = extract_tables
        self.max_tasks_per_child = max(1, max_tasks_per_child)

    def process_pdf_directory_deep(self, input_dir: str, output_dir: str) -> dict:
        """
        Recursively load .pdf and .txt files into one document dictionary.
        """
        input_base = Path(input_dir).resolve()
        output_base = Path(output_dir).resolve()

        if input_base.is_file():
            source_paths = [input_base]
        else:
            source_paths = [path for path in input_base.rglob("*") if path.is_file()]

        # Read existing text directly and build worker tasks only for PDFs.
        tasks = []
        document_dict = {}
        text_file_count = 0
        for source_path in source_paths:
            relative_path = (
                Path(source_path.name)
                if input_base.is_file()
                else source_path.relative_to(input_base)
            )

            # Notebook checkpoints and other hidden folders commonly contain
            # duplicate source files and should not enter the corpus.
            if any(part.startswith(".") for part in relative_path.parts):
                continue

            suffix = source_path.suffix.lower()
            if suffix == ".txt":
                try:
                    text = source_path.read_text(encoding="utf-8-sig")
                    text = strip_metadata_headers(text)
                    document_dict[str(source_path)] = {
                        "extracted_text": text,
                        "output_filename": str(source_path),
                        "md5": _calculate_md5(source_path),
                        "status": "loaded_text_input",
                    }
                    text_file_count += 1
                    print(f"[TEXT] Loaded: {source_path.name}", flush=True)
                except Exception as error:
                    print(
                        f"[TEXT FAILED] {source_path.name}: {error}",
                        flush=True,
                    )
            elif suffix == ".pdf":
                output_filepath = (output_base / relative_path).with_suffix(".txt")
                tasks.append((
                    str(source_path),
                    str(output_filepath),
                    self.x_tolerance,
                    self.y_tolerance,
                    self.extract_tables,
                ))

        if not tasks and not document_dict:
            print("No PDF or text files found.")
            return document_dict

        print(
            f"Loaded {text_file_count} text file(s) directly; "
            f"processing {len(tasks)} PDF file(s) with "
            f"{self.max_workers} worker(s); table extraction "
            f"{'enabled' if self.extract_tables else 'disabled'}."
        )

        def save_result(task):
            pdf_key, result_data = _process_single_pdf_worker(task)
            document_dict[pdf_key] = result_data

        if not tasks:
            pass
        elif self.max_workers == 1:
            # Avoid process-pool and serialization overhead in the safest mode.
            for task in tasks:
                try:
                    save_result(task)
                except Exception as e:
                    print(f"Worker process failed with error: {e}")
        else:
            executor_options = {
                "max_workers": self.max_workers,
                # Spawn workers without inheriting a notebook's already-loaded
                # models and other large in-memory objects.
                "mp_context": get_context("spawn"),
            }
            if sys.version_info >= (3, 11):
                executor_options["max_tasks_per_child"] = self.max_tasks_per_child

            executor = ProcessPoolExecutor(**executor_options)
            futures = {
                executor.submit(_process_single_pdf_worker, task): task
                for task in tasks
            }

            try:
                for future in as_completed(futures):
                    try:
                        pdf_key, result_data = future.result()
                        document_dict[pdf_key] = result_data
                    except Exception as e:
                        failed_pdf = Path(futures[future][0]).name
                        print(f"Worker failed for {failed_pdf}: {e}")
            except KeyboardInterrupt:
                print("\nStopping PDF workers...", flush=True)
                for future in futures:
                    future.cancel()

                # Python 3.11 has no public terminate_workers() method. Capture
                # and terminate active children before non-blocking shutdown so
                # Ctrl+C does not wait for long pdfplumber operations to finish.
                worker_processes = list(
                    getattr(executor, "_processes", {}).values()
                )
                for process in worker_processes:
                    if process.is_alive():
                        process.terminate()

                executor.shutdown(wait=False, cancel_futures=True)
                for process in worker_processes:
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                raise
            else:
                executor.shutdown(wait=True)

        # Stable ordering makes downstream retained-original selection
        # reproducible when deduplicating documents.
        return dict(sorted(document_dict.items()))

def load_pdf(
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    max_workers=2,
    extract_tables=True,
):
    # Instantiate the class
    converter = PdfToTxt(
        x_tolerance=1.5,
        y_tolerance=3,
        max_workers=max_workers,
        extract_tables=extract_tables,
    )
    
    # Process all PDFs recursively
    results = converter.process_pdf_directory_deep(INPUT_FOLDER, OUTPUT_FOLDER)
    
    print(f"\nTotal source files loaded: {len(results)}")
    return results
    
