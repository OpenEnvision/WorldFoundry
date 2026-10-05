# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LlamaGuard S1–S14 unsafe-category labels used by Cosmos prompt checks.

Keys match the official LlamaGuard taxonomy (``S1`` … ``s14``).  Values
are short English labels only; the classifier lives outside this module.
"""

UNSAFE_CATEGORIES = {
    "S1": "Violent Crimes.",
    "S2": "Non-Violent Crimes.",
    "S3": "Sex Crimes.",
    "S4": "Child Exploitation.",
    "S5": "Defamation.",
    "S6": "Specialized Advice.",
    "S7": "Privacy.",
    "S8": "Intellectual Property.",
    "S9": "Indiscriminate Weapons.",
    "S10": "Hate.",
    "S11": "Self-Harm.",
    "S12": "Sexual Content.",
    "S13": "Elections.",
    "s14": "Code Interpreter Abuse.",
}
