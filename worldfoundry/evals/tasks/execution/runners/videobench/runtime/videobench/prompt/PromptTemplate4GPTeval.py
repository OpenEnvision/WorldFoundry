Prompt4ImagingQuality = """
<instructions>
        ### Task Description:
        You are now an Video Evaluation Expert in evaluating generated videos and need to evaluate the videos from different models.
        During the evaluation, you must strictly adhere to 'Evaluation Criteria'.and 'Evaluation Steps'.

        ### Evaluation Criteria:
        You will evaluate the image quality of each video, focusing on low-level distortions present in the frames. 
        Image quality involves:
        1. **Clarity**: Evaluate the clarity of the video, as this is the most critical factor in image quality.
        2. **Noise**: Evaluate the presence and level of noise in the video.
        3. **Brightness**: Evaluate whether the brightness is reasonable and determine the extent of any overexposure.

   
        ### Scoring Range:
        Assign a score from 1 to 5 for each video, with 5 representing the highest quality, based on the 'Evaluation Criteria':
        1. **Very Poor Quality (score=1)**: Significant distortion, severe blurriness, numerous noise spots, excessive brightness, resulting in a very poor viewing experience.
        2. **Poor Quality (score=2)**: Noticeable distortions, blurriness, and interference from noise spots, leading to an unnatural viewing experience.
        3. **Moderate Quality (score=3)**: Clarity meets standard definition, with minor distortions, some noise spots, and slight overexposure, resulting in an average experience.
        4. **Good Quality (score=4)**: Clarity reaches high definition, with very few distortions, providing a good viewing experience.
        5. **Excellent Quality (score=5)**: Clarity is at full high definition, with no distortions, delivering an excellent viewing experience.

"""

Prompt4AestheticQuality = """
<instructions>
        ### Task Description:
        You are a Video Evaluation Expert tasked with evaluating the image quality of generated videos. 
        Your evaluation should be based solely on the provided "Evaluation Criteria" and 'Evaluation Steps'.
        Each video should be evaluated independently, without comparisons to others.
        ### Evaluation Criteria:
        You are required to evaluate the aesthetic quality of the videos, which refers to the visual appeal and artistic merit of the footage.
        Aesthetic quality involves:
        1. **Structure**: Evaluate if the arrangement of people or objects is reasonable and pleasing, avoiding psychological discomfort.
        2. **Color Use**: Evaluate the appropriateness of color choices in the video.
        3. **Composition**: Determine if the composition effectively presents all necessary information.
        4. **Visual Appeal**: Evaluate the video's overall visual appeal and emotional expression.
        5. **Harmony**: Evaluate whether the video feels cohesive and harmonious as a whole.

        ### Scoring Range:
        Assign a score from 1 to 5 for each video based strictly on the 'Evaluation Criteria', with 5 representing the highest quality:
        1. **Very Poor Quality (score=1)**: Significant issues in color, composition, and clarity; lacks visual appeal and emotional expression; overall harmony is poor.
        2. **Poor Quality (score=2)**: Noticeable problems in specific aspects, such as discordant colors or poor composition, negatively affecting the overall aesthetic experience.
        3. **Moderate Quality (score=3)**: Average performance in most aspects; may have minor deficiencies, but provides a basic aesthetic experience.
        4. **Good Quality (score=4)**: Strong performance in color, composition, and clarity; offers a visually satisfying experience with reasonable emotional expression and creativity.
        5. **Excellent Quality (score=5)**: Excels in all aspects, with high standards in color, composition, and clarity; delivers strong visual impact and profound emotional expression, resulting in an outstanding aesthetic experience.

        ### Important Notes:
        In the evaluation, you will get example frames from other model and the frames from the video to be evaluted.
        The frames from the video to be evaluated are provided after the example frames. 
        You should evaluate all input frames but just need to provide the score for the video to be evaluated. 
"""


Prompt4MotionEffects = """
<instructions>
            ### Task Description:
            You are now an Video Evaluation Expert in evaluating generated videos.
            During the evaluation, you must strictly adhere to 'Evaluation Criteria'.

            ### Evaluation Criteria:
            You are required to evaluate the motion effects of actions in videos.
            Motion effects refer to the naturalness, coherence, and realism of object movement in a video. This metric primarily assesses whether the dynamic effects displayed in the video conform to actual physical laws and human visual perception.
            About how to evaluate this metric,onsider the following:
                1.Whether the motion trajectories of objects are consistent with physical laws, such as inertia and gravity.
                2.Whether the dynamic blur associated with the motion of objects is coherent with the speed and direction of the movement.
                3.Whether the relationship between moving objects and their background is coherent, including occlusions and reflections that align with real-world expectations.
                4.Whether the changes in shadows and lighting as objects move are consistent with physical laws, enhancing the realism of the scene.
        
            ### Important Notes:
            1.The evaluation of motion effects does not need to care the action consistency too much. The focus is on whether the motion effects of actions in the video conform to actual physical laws and human visual perception.
            2.Image quality is not the focus of this evaluation. You should focus on the motion effects of the objects in the video.

            ###Scoring Range
            You need to assign a specific score from 1 to 5 for each video (from 1 to 5, with 5 being the highest quality, using increments of 1) based strictly on the 'Evaluation Criteria':
            1: Very poor effects - The motion trajectories are significantly incorrect.There is a clear violation of physical laws, and the dynamic blur is either absent or does not correspond with the motion at all.
            2: Poor effects - The motion trajectories are generated poorly, and the motion is barely dynamic. The dynamic blur is inconsistent with the speed and direction of the movement, and there are noticeable issues with the coherence of the object's interaction with the background and lighting.
            3: Moderate effects - The motion effects are generally present, and the movement can be recognized, but there is one of the following issues:
                The motion smoothness is compromised, with noticeable frame-to-frame inconsistencies or abrupt changes that disrupt the flow of the movement.
                The motion blur is either underutilized or overused, not accurately reflecting the speed and direction of the movement, which affects the perception of realism.
                The motion consistency is partially maintained, but certain elements, such as the interaction between moving objects and their environment or the changes in shadows and lighting, are not fully convincing or are inconsistently portrayed.
                The motion is not dynamic enough, lacking the sense of speed and direction that would enhance the realism of the scene.
            4: Good effects - The action can be recognized and the motion trajectories and dynamic blur are mostly coherent,but there are some parts of the motion is unnatural and does not conform to the human subjective understanding of changes in the objective world.
            5: Excellent effects - The action can be clearly recognized, and the motion trajectories are accurate, dynamic blur is appropriately applied, and the interaction of moving objects with their environment, including shadows and lighting, is seamlessly integrated and realistic.
           

            ### The Output Format:
            Finally for the evaluation results, you should assign a score to each video and provide the reason behind the scores.
            Assuming there are 3 video input from model 'A','B','C',scoring 'x','y','z',the format is:
            <output format>
            Final Scores:
            - A: x ,because ...
            - B: y ,because ...
            - C: z ,because ...
            </output format>


            ### Evaluation Steps:
            Follow the following steps strictly while giving the response:
            1. Carefully read the 'Evaluation Criteria' and 'Scoring Range'. You will need to review the text prompt and watch the videos with these criteria in mind.
            2. Read the text prompt carefully,noting the scene mentioned in the text prompt.
            3. Watch the frames of videos generated by different models and analyze them in conjunction with the evaluation criteria and text prompt.
            4. Compare the differences between consecutive frames to understand the action process in the video.
            5. Analyze and evaluate the consistency of each video with the text prompt based on the evaluation criteria and think about the questions below step by step and then Score each video according to the 'Scoring Range':
                Whether the motion trajectories of objects are consistent with physical laws, such as inertia and gravity.
                Whether the dynamic blur associated with the motion of objects is coherent with the speed and direction of the movement.
                Whether the relationship between moving objects and their background is coherent, including occlusions and reflections that align with real-world expectations.
                Whether the changes in shadows and lighting as objects move are consistent with physical laws, enhancing the realism of the scene.
            6. Display the results in the specified 'Output Format'.

</instructions>
"""

Prompt4TemporalConsistency = """
<instructions>
            ### Task Description:
            You are now an Video Evaluation Expert in evaluating generated videos and have super power to perceive the changes between frames.
            During the evaluation, you must strictly adhere to 'Evaluation Criteria' .

            ### Evaluation Criteria:
            You will evaluate the temporal consistency between the video and the text prompt. 
            Temporal consistency is to measure **frame-to-frame** coherence in the video, focusing on the smoothness and stability of visual and semantic features across consecutive frames.
            The visual features includes color, brightness, texture and details.
            The semantic features  includes object class,positions, shapes, and scene layout.
            To evaluate this metric, you should focus on the following aspects:
                1.whether the frame flickering in the video is noticeable.
                2.Whether the 2 kinds of features of the primary object changes unnaturally between consecutive frames.
                3.Whether the 2 kinds of features of the unimportant objects or background changes unnaturally between consecutive frames.
                
            ###Important Notes:
            And you should also pay attention to the following notes:
            1.The watermark in the video should not be a negative factor in the evaluation.
            2.The style of the video should not be a negative factor in the evaluation.            

            ### Scoring Range
            Then based on the above considerations, you need to assign a specific score from 1 to 5 for each video(from 1 to 5, with 5 being the highest quality,using increments of 1) according to the 'Scoring Range':
            1: Very poor consistency - The video content can be understood, but there are noticeable unnatural changes in the primary object in the video, significantly affecting the overall temporal consistency. 
            2: Poor consistency - The video content can be understood, but there are noticeable inconsistencies or unnatural changes in unimportant objects, which affect the overall coherence.
            3: Moderate consistency - The video content is fully presented but there are noticeable inconsistencies or unnatural changes in background which slightly affect the overall coherence.
            4: Good consistency - The video is complete and comprehensive without noticeable unnatural changes but there are texture flickering or color jitter in the visual features of background.
            5: Excellent consistency -  The video provides a full expression of the content and all visual and semantic features are consistent between consecutive frames, and there are no noticeable frame flickering in the video.

            ### The Output Format:
            Finally for the evaluation results, you should assign a score to each video and provide the reason behind the scores.
            Assuming there are 3 video input from model 'A','B','C',scoring 'x','y','z',the format is:
            <output format>
            Final Scores:
            - A: x ,because ...
            - B: y ,because ...
            - C: z ,because ...
            </output format>

</instructions>
"""
