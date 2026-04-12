from __future__ import annotations 



from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT.parent / "data" / "ViLStrUB"
CATEGORY_ORDER = ["vp", "pp", "anaph", "ellip", "adj", "vb", "conj"]
CATEGORY_DATASET_CONFIG = {
    "vp": {
        "json_path": DATA_ROOT / "jsons_UNIT2" / "vp.json",
        "image_dir": DATA_ROOT / "images" / "vp",
    },
    "pp": {
        "json_path": DATA_ROOT / "jsons_UNIT2" / "pp.json",
        "image_dir": DATA_ROOT / "images" / "pp",
    },
    "anaph": {
        "json_path": DATA_ROOT / "jsons_UNIT2" / "anaph.json",
        "image_dir": DATA_ROOT / "images" / "anaph",
    },
    "ellip": {
        "json_path": DATA_ROOT / "jsons_UNIT2" / "ellip.json",
        "image_dir": DATA_ROOT / "images" / "ellip",
    },
    "adj": {
        "json_path": DATA_ROOT / "jsons_UNIT2" / "adjscope.json",
        "image_dir": DATA_ROOT / "images" / "adjscope",
    },
    "vb": {
        "json_path": DATA_ROOT / "jsons_UNIT2" / "verbscope.json",
        "image_dir": DATA_ROOT / "images" / "verbscope",
    },
    "conj": {
        "json_path": DATA_ROOT / "jsons_UNIT2" / "conj.json",
        "image_dir": DATA_ROOT / "images" / "conj",
    },
}


VS_JSON_PATH = "../../../../data/ViLStrUB/jsons_UNIT2/"
VS_IMAGE_PATH = "../../../../data/ViLStrUB/images/"
# It would be better if I could treat them as environment variables I guess 

SIMPLE_SELECTION_SYSTEM_PROMPT = """
You are a vision-language reasoning model performing structural disambiguation using visual context.

You will be given:
- an ambiguous caption
- an image

Your task is to select the correct interpretation of the caption based on the image.

Important:
- All options are derived from the same ambiguous caption and differ only in structural interpretation.
- You MUST use the image to decide.
- Do NOT rely only on textual plausibility.
"""

SIMPLE_SELECTION_USER_PROMPT_CAPTION_AND_IMAGE = """
Caption: "{input_caption}"

And now, here is the image
"""

SIMPLE_SELECTION_USER_PROMPT_QUESTION_TWO_MEANINGS = """  
Among the following options, which one is the right disambiguated meaning of the caption with the help of the image?
1) "{resolved_caption_correct}"
2) "{resolved_caption_incorrect}"
3) I cannot decide
""" 

SIMPLE_SELECTION_USER_PROMPT_QUESTION_THREE_MEANINGS = """
Among the following options, which one is the right disambiguated meaning of the caption with the help of the image?
1) "{resolved_caption_correct}"
2) "{resolved_caption_incorrect_1}"
3) "{resolved_caption_incorrect_2}"
4) I cannot decide
"""

SIMPLE_SELECTION_USER_PROMPT_CONCLUDE = """
Please provide your answer in the following format:
Answer: <one of the options> as a single number (1, 2, 3, etc.) 
"""

STEP1_SYSTEM_PROMPT_TASK_INTRODUCTION = """
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

STEP1_USER_PROMPT_CAPTION_AND_IMAGE = """
Caption: "{input_caption}"

And now, here is the image
"""

# for those cases where resolved meanings are two 
STEP1_USER_PROMPT_QUESTION_TWO_MEANINGS = """  
Among the following options, which one best describes the relationship between the caption and the image?
1) The caption is ambiguous, and with the help of the image, the resolved meaning is "{resolved_caption_correct}"
2) The caption is ambiguous, and with the help of the image, the resolved meaning is "{resolved_caption_incorrect}"
3) The caption is already disambiguated and matches the image
4) The caption is already disambiguated but does not match the image, the right meaning is "{the_other_caption}"
5) I cannot decide  
"""

# for those cases where resolved meaning counts up to three 
STEP1_USER_PROMPT_QUESTION_THREE_MEANINGS = """
Among the following options, which one best describes the relationship between the caption and the image?
1) The caption is ambiguous, and with the help of the image, the resolved meaning is "{resolved_caption_correct}"
2) The caption is ambiguous, and with the help of the image, the resolved meaning is "{resolved_caption_incorrect_1}"
3) The caption is ambiguous, and with the help of the image, the resolved meaning is "{resolved_caption_incorrect_2}"
4) The caption is already disambiguated and matches the image
5) The caption is already disambiguated but does not match the image, the right meaning is "{the_other_caption}"
6) The caption is already disambiguated but does not match the image, the right meaning is "{the_other_caption_2}"
7) I cannot decide  
"""

STEP1_USER_PROMPT_CONCLUDE = """
Please provide your answer in the following format:
Answer: <one of the options> as a single number (1, 2, 3, etc.) 
Explanation: <a short explanation of your reasoning in one sentence>
"""


STEP2_SYSTEM_PROMPT_TASK_INTRODUCTION_WITH_CONTEXT = """
You are correct! Now, I will give you another image with the same caption, but this time, the image will be different from the one you just saw.
You are going to do the same task as before, which is to determine the relationship between the caption and the new image by picking one of the multiple options provided by the question.
"""

STEP2_SYSTEM_PROMPT_TASK_INTRODUCTION_WITHOUT_CONTEXT = """
Now, I will give you another image with the same caption, but this time, the image will be different from the one you just saw.
You are going to do the same task as before, which is to determine the relationship between the caption and the new image by picking one of the multiple options provided by the question.
"""

STEP2_USER_PROMPT_CAPTION_AND_IMAGE = """
Caption: "{input_caption}"
And now, here is the new image
"""


STEP2_USER_PROMPT_QUESTION_TWO_MEANINGS = """  
Among the following options, which one best describes the relationship between the caption and the new image?
1) The caption is ambiguous, and with the help of the image, the resolved meaning is "{resolved_caption_correct}"
2) The caption is ambiguous, and with the help of the image, the resolved meaning is "{resolved_caption_incorrect}"
3) The caption is already disambiguated and matches the image
4) The caption is already disambiguated but does not match the image, the right meaning is "{the_other_caption}"
5) I cannot decide  
"""


STEP2_USER_PROMPT_QUESTION_THREE_MEANINGS = """
Among the following options, which one best describes the relationship between the caption and the new image?
1) The caption is ambiguous, and with the help of the image, the resolved meaning is "{resolved_caption_correct}"
2) The caption is ambiguous, and with the help of the image, the resolved meaning is "{resolved_caption_incorrect_1}"
3) The caption is ambiguous, and with the help of the image, the resolved meaning is "{resolved_caption_incorrect_2}"
4) The caption is already disambiguated and matches the image
5) The caption is already disambiguated but does not match the image, the right meaning is "{the_other_caption}"
6) The caption is already disambiguated but does not match the image, the right meaning is "{the_other_caption_2}"
7) I cannot decide  
"""

STEP2_USER_PROMPT_CONCLUDE = """
Please provide your answer in the following format:
Answer: <one of the options> as a single number (1, 2, 3, etc.) 
Explanation: <a short explanation of your reasoning in one sentence>
"""