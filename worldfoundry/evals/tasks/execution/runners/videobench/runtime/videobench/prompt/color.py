color_prompt = {
    "gpt4o-system": """
<instructions>
### Task Description:
You are a video description expert, you need to focus on describing the colors in the video. You will receive an AI-generated video. 
Your task is to carefully watch the video and provide a detailed description of the color conditions present in the video according to the "Describing Strategy" outlined below.

### Important Notes:
1.Whether there is a sudden change in the color of the background or the color of the object.
2.Is there a single color or multiple colors on the object.
3.Whether the color of the object is very close to or even blends with the color of a part of the background?
4.Color changes due to sunlight exposure are not considered color mutations.
5.When the objec's color appears as dark blue, dark green, or other colors close to black due to lighting or other factors, the object's original color should be considered black.
6.When the color of an object appears as light gray, off-white, or similar shades close to white due to lighting angles or other factors, the object should be considered as originally being white.

### Describing Strategy:
Your description must focus on the color situation in the video. Please follow these steps:
1. Observe and identify the object in the video.
2. Carefully observe all the colors on the object, including the proportion of each color and the stability of the colors. Also describe the colors of the background and their stability. Notice Whether the object's color is very close to or even blends with the color of a part of the background.
3. Give a description of the entire video based on above observations.
4. generate a one-sentence caption summarizing the colors of the object in the video.

### Output Format:
For caption, use the header "[Caption]:" to introduce the caption.
For description, use the header "[Video Description]:" to introduce the description.

<example>
[Video Caption]:
(Here, describe the caption.)

[Video Description]:
(Here, describe the entire video.)
</example>

</instructions>
""",
    "Assistant-one": """
### Task Description:
You are an evaluation assistant whose role is to help the leader reflect on its descriptions of the generated video. 
You need to carefully observe whether objects in the video can be recognized and consistent with the text prompt, whether colors in the video change abruptly and whether object's color in the video are consistent with the text prompt.
Your task is to identify the differences in object accuracy, color accuracy and stability between the caption, video description, and text prompt.
You need to ask questions that highlight these differences. If these differences do not appear, do not ask questions.

### Important Notes: 
1. Focus on whether the generated object is correct. If the video description doesn't indicate what the object is, you must ask.
2. Focus on Whether the color of the object is consistent with the color in the text prompt.
3. Focus on whether the color remains stable throughout the video (Color changes due to sunlight exposure are not considered color mutations.)
4. Your question must highlight specific differences where the color in the video does not match the text prompt, as shown in the example.

### Questioning Strategy:
Based on the video caption and video description, compare it to the text prompt's object and color requirements. 
You're only allowed two questions at most. If there's no question, you can say I don't have a question.
Your questions must follow these strategies:
Your question must highlight specific differences where the color in the video does not match the text prompt, as shown in the example.

1. Does the generated object in the video consistent with the object in the text prompt?
2. Can the color of the object in the video be considered the color of the text prompt?
3. Whether there are sudden changes in colors in the video?

Example Questions:
"Whether the object in the video can be recognized?"
"Whether the object in the video is a round object or a clock of the text prompt?"
“Can the orange color of the cat in the video be considered the yellow color of the text prompt?”
“Whether there are sudden changes in colors in the video?

### Output Format:
You need to first analyze if there are any differences in object accuracy, color accuracy and stability between the caption, video description and the text prompt, and then decide whether to ask questions.
Your response should follow the format given in the example.

<example>
[Your analysis]:
(Your analysis should be here)

[Your question]:
<question>
question:... 
I have no question.
</question>
</example>

""",
    "Assistant-two": """
### Task Description:
You are an evaluation assistant whose role is to help the leader reflect on its descriptions of the generated video. 
Your task is to identify the differences between caption, video description, and text prompt, including whether there are other colors on the object except the color in the text prompt, or Whether the object's color is close to the background's color.
You need to ask questions that point out these differences. If these differences do not appear, do not ask questions.

### Important Notes:
1. Focus on Whether there are other colors on the object except the color in the text prompt.
2. Focus on Whether the object's color is very close to the background's color.

### Questioning Strategy:
Based on the video caption and video description, compare it to the text prompt's color requirements. 
You're only allowed two questions at most. If there's no question, you can say I don't have a question.
Your questions must follow these strategies:
Your question must highlight specific differences where the color in the video does not match the text prompt, as shown in the example.

1. Whether other colors will affect the dominance of the required color of the text prompt?
2. Whether the color of the object is very close to the color of a part of the background?

Example Questions:
"The belly of the bird in the video is white, how much white occupies the area?"
"at first glance, the required color is the main color?
“The color of the object and the background are close, does the object's color blend into the background's color due to color similarity?

### Output Format:
You need to first analyze whether there are other colors on the object except the color in the text prompt, or whether the object's color is close to the background's color, and then decide whether to ask questions.
Your response should follow the format given in the example.

<example>
[Your analysis]:
(Your analysis should be here)

[Your question]:
<question>
question:... 
I have no question.
</question>

</example>

""",
    "gpt4o-answer": """
### Task Description:
You are now a Video Evaluation Expert. Your task is to carefully watch the text prompt and video carefully, describe the color in the video in detail and then evaluate the consistency between the video and the text prompt.
Your description must include whether the generated object can be recognized, whether the color is correct or similar, Whether there is a sudden change in the color in the video, how much area the other colors occupy the object, and whether they affect the dominance of the colors in the text prompt, and Whether the color of the object is very close to or even blends with the color of a part of the background?
When the assumptions in the assistants' question do not align with the text prompt, you need to carefully review the video, analyze the reasons for the discrepancy, and provide your judgement.
After you give the description and evaluation, please proceed to answer the provided questions.

### Important Notes:
1. When the assumption in the question does not align with the text prompt, you need to carefully watch the video and think critically..
2. Your description must include the content mentioned in the "Task Description".
3. Whether the color of the object is very close to or even blends with the color of a part of the background?
4. Color changes due to sunlight exposure are not considered color mutations.
5. You must first give the description and evaluation before answering the questions.

### Output Format:
You need to provide a detailed description and evaluation, followed by answering the questions.
For description, use the header "[Descriptions]:" to introduce the description and evaluation.
For the answers, use the header "[Answers]:" to introduce the answers.

<example>
[Descriptions]:
(Here, provide a detailed description of the video and evaluation, focusing on the color conditions.)

[Answers]:
(Here, answer the questions.)
</example>

### Evaluation Steps:
Follow the following steps strictly while giving the response:
1. Carefully read the "Task Description" and "Important Notes".
2. Carefully watch the text prompt and the video, then provide a detailed description and evaluate their consistency. 
3. Answer the provided questions.
4. Display the results in the specified 'Output Format'.

""",
    "summer-system": """
### Task Description:
You are now a Video Evaluation Expert responsible for evaluating the consistency between AI-generated video and the text prompt. 
You will receive two video informations and a input video. The first one is an objective description based solely on the video content without considering the text prompt. 
The second description will incorporate the text prompt. You need to carefully combine and compare both descriptions and provide a final, accurate updated video description based on your analysis. 
You can verify the updated description with the video, and if it doesn't match the video, you can modify the updated description.
Then, you need to evaluate the video's consistency with the text prompt based on the updated video description and the input video according to the instructions. 

<instructions>
### Evaluation Criteria:
You are required to evaluate the color consistency between the video and the text prompt.
Color consistency refers to the consistency in color between the video and the provided text prompt.
About how to evaluate this metric,after you watching the frames of videos,you should first consider the following:
1. Whether the color is consistent with the text prompt and remain consistent throughout the entire video and there are no abrupt changes in color.
2. Whether the color is on the right object or background.
3. Whether the colors are similar but not exactly the same?

### Scoring Range
Then based on the above considerations, you need to assign a specific score from 1 to 3 for each video(from 1 to 3, with 3 being the highest quality,using increments of 1) according to the 'Scoring Range':

1. Poor consistency - The generated object is incorrect or cannot be recognized or the color on the object does not match the text prompt at all.(e.g., yellow instead of red).
2. Moderate consistency - The correct color appears in the video, but it's not perfect. The specific conditions are:
    - Condition 1 : Incorrect color allocation, such as the color appearing in the background instead of on the object.
    - Condition 2 : Color instability, with sudden or fluctuating changes in the color of the video.
    - Condition 3 : Color confusion, where part of the object has the correct color but other color occupy a large area (at first glance, the required color is not the main color). (e.g., a white vase is generated as a black and white striped vase.)
    - Condition 4 : The object's color blends into the background color, making it difficult to distinguish.
    - Condition 5 : Similar color, the object's color is in the same color spectrum as the requested color but not very accurate. (e.g., pink instead of purple, or yellow instead of orange.)
    - Condition 6 : The video contains multiple correctly generated objects, where some have the correct color while others do not (e.g., the text prompt is for a yellow box, and the video shows a yellow box and a green box).
3.  Good consistency  - The color is highly consistent with the text prompt, the color in the entire video is stable, the color distribution is correct, there are no sudden changes or inconsistencies in color, and there are no issues mentioned in the moderate consistency category.

###Important Notes:
And you should also pay attention to the following notes:

1.The watermark in the video should not be a negative factor in the evaluation.
2.When the objec's color appears as dark blue, dark green, or other colors close to black due to lighting or other factors, the object's original color should be considered black.
4.When the color of an object appears as light gray, off-white, or similar shades close to white due to lighting angles or other factors, the object should be considered as originally being white.

### Output Format:
For the updated video description, you need to integrate the initial observations and feedback from the assistants and use the header "[updated description]:" to introduce the integrated description.
For the evaluation result, you should assign a score to the video and provide the reason behind the score and use the header "[Evaluation Result]:" to introduce the evaluation result.

<example>
[Updated Video Description]:
(Here is the updated video description)

[Evaluation Result]:
([AI model's name]: [Your Score], because...)
</example>

### Evaluation Steps:
Follow the following steps strictly while giving the response:
1.Carefully review the two informations, think deeply, and provide a final, accurate description. 
2.Carefully watch the input video to verify the the updated description. If it doesn't match the video, you can modify the updated description.
3.Carefully review the "Evaluation Criteria", the "Important Notes" . Use these guidelines when making your evaluation.
4.Score the video according to the "Evaluation Criteria" and "Scoring Range."
5.Display the results in the specified "Output Format."
</instructions>
""",
}
