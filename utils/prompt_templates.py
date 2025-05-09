# 用于检索重排的提示模板
RERANKING_PROMPT = """
Please rank the following paragraphs based on their relevance to the query.
The most relevant paragraphs should directly address the claims in the query.

Query: "{query}"

Paragraphs:
{paragraphs}

Return ONLY the numbers of the top {top_k} most relevant paragraphs in order, separated by commas.
For example: 3,1,5,2,4
"""

# 用于初步验证的提示模板
INITIAL_VERIFICATION_PROMPT = """
I need you to initially analyze the following claim based on the provided media content. 
Don't make a final judgment yet, just provide your initial thoughts.

Claim: "{claim}"

Media Content: "{media_text}"

Provide your initial analysis about whether the claim appears to be true, false, or partially true 
based on the media content alone. If you're uncertain, explain why.

Format your response as JSON with the following fields:
- initial_judgment: "true", "false", "partially_true", or "uncertain"
- confidence: a number between 0 and 1
- reasoning: your explanation for the initial judgment
"""

# 用于基于证据的验证的提示模板
EVIDENCE_VERIFICATION_PROMPT = """
I need you to verify the following claim based on the provided media content and evidence.

Claim: "{claim}"

Media Content: "{media_text}"

Evidence:
{evidence}

Important: Even if you had an initial impression about the claim, your judgment must be based on the provided evidence. Remember that any identified people in the media might be incorrect, so verify their identities using the evidence.

Analyze each piece of evidence and determine whether it supports, contradicts, or is neutral toward the claim.

Format your response as JSON with the following fields:
- final_judgment: "true", "false", "partially_true", or "uncertain"
- confidence: a number between 0 and 1
- reasoning: your detailed explanation of the judgment
- evidence_analysis: an array of objects, each containing:
    - evidence_id: the number of the evidence [n]
    - relevance: "supporting", "contradicting", or "neutral"
    - explanation: why this evidence supports/contradicts/is neutral to the claim
"""

# 用于claim分解的提示模板
CLAIM_DECOMPOSITION_PROMPT = """
Please break down the following claim into simple subject-predicate-object sentences.
Each sentence should express only one basic fact.

Claim: "{claim}"

Output each simple sentence on a new line, without numbering.
"""