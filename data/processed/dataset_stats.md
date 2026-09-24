Total rows: 4910

|                              |    n |
|:-----------------------------|-----:|
| ('agentdojo', 'attack')      | 2450 |
| ('agentdojo', 'benign')      |  142 |
| ('hard_negatives', 'benign') |   40 |
| ('injecagent', 'attack')     | 2108 |
| ('injecagent', 'benign')     |  170 |

|                                                   |   n |
|:--------------------------------------------------|----:|
| ('agentdojo', 'attack', 'direct')                 | 490 |
| ('agentdojo', 'attack', 'ignore_previous')        | 490 |
| ('agentdojo', 'attack', 'important_instructions') | 490 |
| ('agentdojo', 'attack', 'injecagent')             | 490 |
| ('agentdojo', 'attack', 'tool_knowledge')         | 490 |
| ('agentdojo', 'benign', 'none')                   | 142 |
| ('hard_negatives', 'benign', 'none')              |  40 |
| ('injecagent', 'attack', 'Data Security Harm')    | 374 |
| ('injecagent', 'attack', 'Financial Data')        | 204 |
| ('injecagent', 'attack', 'Financial Harm')        | 306 |
| ('injecagent', 'attack', 'Others')                | 510 |
| ('injecagent', 'attack', 'Physical Data')         | 374 |
| ('injecagent', 'attack', 'Physical Harm')         | 340 |
| ('injecagent', 'benign', 'none')                  | 170 |

|                                         |   n |
|:----------------------------------------|----:|
| ('attack', 'base', 'data_stealing')     | 544 |
| ('attack', 'base', 'direct_harm')       | 510 |
| ('attack', 'enhanced', 'data_stealing') | 544 |
| ('attack', 'enhanced', 'direct_harm')   | 510 |
| ('benign', nan, nan)                    | 170 |

Token lengths (tokenizer: protectai/deberta-v3-base-prompt-injection-v2, incl. special tokens)
|                              |   count |   mean |    std |   min |   50% |   90% |    99% |   max |
|:-----------------------------|--------:|-------:|-------:|------:|------:|------:|-------:|------:|
| ('agentdojo', 'attack')      |    2450 |  597.6 | 1165.4 |    31 | 339   | 948.4 | 7222.5 |  8129 |
| ('agentdojo', 'benign')      |     142 |  168.1 |  579.1 |     3 |  76   | 295.9 |  929.2 |  6741 |
| ('hard_negatives', 'benign') |      40 |   86.1 |   21.2 |    54 |  82   | 117.1 |  139   |   141 |
| ('injecagent', 'attack')     |    2108 |  111.7 |   35.6 |    18 | 107   | 163   |  197   |   230 |
| ('injecagent', 'benign')     |     170 |   90   |   32.8 |    17 |  79.5 | 144   |  158.6 |   164 |

AgentDojo token lengths by `paired` (benign paired = clean version of an output that gets injected):
|                   |   count |   mean |    std |   min |   50% |   90% |   max |
|:------------------|--------:|-------:|-------:|------:|------:|------:|------:|
| ('attack', True)  |    2450 |  597.6 | 1165.4 |    31 |   339 | 948.4 |  8129 |
| ('benign', False) |      93 |   60.3 |   70.7 |     3 |    33 | 108.6 |   491 |
| ('benign', True)  |      49 |  372.5 |  954.2 |    12 |   194 | 627.4 |  6741 |

Rows over 512 tokens:
|                              |   sum |   mean |
|:-----------------------------|------:|-------:|
| ('agentdojo', 'attack')      |   673 |  0.275 |
| ('agentdojo', 'benign')      |     8 |  0.056 |
| ('hard_negatives', 'benign') |     0 |  0     |
| ('injecagent', 'attack')     |     0 |  0     |
| ('injecagent', 'benign')     |     0 |  0     |
