# Frozen Prompt Comparison

Automatic verdicts use only exact frozen expected-concept coverage. Human rubric review remains pending and must be performed blind to model identity.

## Summary

| Group | Prompts | Base mean coverage | CPT mean coverage | Improved | Retained | Degraded | Degradation rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| General | 25 | 0.643 | 0.623 | 2 | 21 | 2 | 8.0% |
| Domain | 25 | 0.055 | 0.041 | 0 | 24 | 1 | 4.0% |

## Assignment Examples

| Prompt ID | Group | Prompt | Base output | CPT output | Automatic verdict |
|---|---|---|---|---|---|
| general-factual-01 | general | The capital of France is | Paris. The capital of the United States is Washington, D.C. The capital of the United Kingdom is London. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of New Zealand is Wellington. The capital of Indi... | Paris. The capital of the United States is Washington, D.C. The capital of the United Kingdom is London. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of New Zealand is Wellington. The capital of Indi... | Retained |
| general-science-01 | general | Water boils at what temperature in degrees Celsius at standard atmospheric pressure? | The boiling point of water is 100 degrees Celsius at standard atmospheric pressure. ### What is the boiling point of water in degrees Celsius? The boiling point of water is 100 degrees Celsius at standard atmospheric pressure. ### What i... | The boiling point of water at standard atmospheric pressure is 100 degrees Celsius. ### What is the boiling point of water in Fahrenheit? The boiling point of water in Fahrenheit is 212 degrees Fahrenheit. ### What is the boiling point o... | Retained |
| general-science-02 | general | The speed of light in vacuum is approximately | 300,000,000 metres per second (186,000 miles per second). ## What is the speed of light in a vacuum? The speed of light in a vacuum is approximately 300,000,000 metres per second (186,000 miles per second). ## What is the speed of light ... | 300,000,000 metres per second (186,000 miles per second). ## What is the speed of light in a vacuum? The speed of light in a vacuum is approximately 300,000,000 metres per second (186,000 miles per second). ## What is the speed of light ... | Retained |
| cisco-switching-02 | domain | Why does a switch need a different bridge ID for each VLAN when using PVST+ or Rapid PVST+? | A. The switch needs a different bridge ID for each VLAN because the switch needs to be able to distinguish between the VLANs. B. The switch needs a different bridge ID for each VLAN because the switch needs to be able to distinguish betw... | A. The switch needs a different bridge ID for each VLAN because the switch needs to be able to distinguish between the VLANs. B. The switch needs a different bridge ID for each VLAN because the switch needs to be able to distinguish betw... | Retained |
| cisco-routing-01 | domain | What type of routing protocol is OSPF, and what information does each OSPF router advertise? | OSPF is a link-state routing protocol. Each OSPF router advertises the following information: - Link-state advertisements (LSAs) - Link-state database (LSDB) - Link-state database (LSDB) summarization - Link-state database (LSDB) summari... | OSPF is a link-state routing protocol. Each OSPF router advertises the following information: - Link-state advertisements (LSAs) - Link-state database - Link-state database summarization - Link-state database summarization - Link-state d... | Retained |
| cisco-datacenter-01 | domain | What are the packet-processing roles of ingress and egress VTEPs in a VXLAN network? | A. The ingress VTEP is responsible for receiving the VXLAN packets and encapsulating them into the VXLAN header. B. The egress VTEP is responsible for receiving the VXLAN packets and encapsulating them into the VXLAN header. C. The ingre... | A. The ingress VTEP is responsible for packet processing and the egress VTEP is responsible for packet processing. B. The ingress VTEP is responsible for packet processing and the egress VTEP is responsible for packet forwarding. C. The ... | Retained |
