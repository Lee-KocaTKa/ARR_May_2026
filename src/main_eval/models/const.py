


VS_JSON_PATH = "../../../../data/ViLStrUB/jsons_UNIT2/"
VS_IMAGE_PATH = "../../../../data/ViLStrUB/images/"
# It would be better if I could treat them as environment variables I guess 

STEP1_SYSTEM_PROMPT = """
You are a vision and language reasoning model specialised in resolving structural ambiguity using visual context.

You are going to be given a pair of an image and a caption, which will be either of the three cases:
1) Ambiguous caption with a clarifying image (the image resolves the ambiguity in the caption)
2) Disambiguated caption with the right image (the caption is already disambiguated and matches the image)
3) Disambiguated caption with the wrong image (the caption is already disambiguated but does not match the image) 

You will be asked to determine the relationship between the caption and the image by picking one of the multiple options provided by the question.

Your answer should be in the following format:
Answer: <one of the options> as a single number (1, 2, 3, etc.) 
Explanation: <a short explanation of your reasoning in one sentence>
"""

STEP1_USER_PROMPT = """
Caption: 
"{caption}"


***