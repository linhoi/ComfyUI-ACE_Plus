all_examples = [
            {
                "input_image": None,
                "input_mask": None,
                "input_reference_image": "assets/samples/portrait/human_1.jpg",
                "save_path": "examples/outputs/portrait_human_1.jpg",
                "instruction": "Maintain the facial features, A girl is wearing a neat police uniform and sporting a badge. She is smiling with a friendly and confident demeanor. The background is blurred, featuring a cartoon logo.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 4194866942,
                "repainting_scale": 1.0,
                "task_type": "portrait",
                "edit_type": "repainting"
            },
            {
                "input_image": None,
                "input_mask": None,
                "input_reference_image": "assets/samples/subject/subject_1.jpg",
                "save_path": "examples/outputs/subject_subject_1.jpg",
                "instruction": "Display the logo in a minimalist style printed in white on a matte black ceramic coffee mug, alongside a steaming cup of coffee on a cozy cafe table.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 2935362780,
                "repainting_scale": 1.0,
                "task_type": "subject",
                "edit_type": "repainting"
            },
            {
                "input_image": "assets/samples/local/local_1.webp",
                "input_mask":  "assets/samples/local/local_1_m.webp",
                "input_reference_image": None,
                "save_path": "examples/outputs/local_local_1.jpg",
                "instruction": "By referencing the mask, restore a partial image from the doodle {image} that aligns with the textual explanation: \"1 white old owl\".",
                "output_h": -1,
                "output_w": -1,
                "seed": 1159797084,
                "repainting_scale": 0.5,
                "task_type": "local_editing",
                "edit_type": "contour_repainting"
            },
            {
                "input_image": "assets/samples/application/photo_editing/1_1_edit.png",
                "input_mask": "assets/samples/application/photo_editing/1_1_m.png",
                "input_reference_image": "assets/samples/application/photo_editing/1_ref.png",
                "save_path": "examples/outputs/photo_editing_1.jpg",
                "instruction": "The item is put on the ground.",
                "output_h": -1,
                "output_w": -1,
                "seed": 2072028954,
                "repainting_scale": 1.0,
                "task_type": "subject",
                "edit_type": "repainting"
            },
            {
                "input_image": "assets/samples/application/logo_paste/1_1_edit.png",
                "input_mask": "assets/samples/application/logo_paste/1_1_m.png",
                "input_reference_image": "assets/samples/application/logo_paste/1_ref.png",
                "save_path": "examples/outputs/logo_paste_1.jpg",
                "instruction": "The logo is printed on the headphones.",
                "output_h": -1,
                "output_w": -1,
                "seed": 934582264,
                "repainting_scale": 1.0,
                "task_type": "subject",
                "edit_type": "repainting"
            },
            {
                "input_image": "assets/samples/application/movie_poster/1_1_edit.png",
                "input_mask": "assets/samples/application/movie_poster/1_1_m.png",
                "input_reference_image": "assets/samples/application/movie_poster/1_ref.png",
                "save_path": "examples/outputs/movie_poster_1.jpg",
                "instruction": "The man is facing the camera and is smiling.",
                "output_h": -1,
                "output_w": -1,
                "seed": 988183236,
                "repainting_scale": 1.0,
                "task_type": "portrait",
                "edit_type": "repainting"
            }

        ]

fft_examples =  [
            {
                "input_image": None,
                "input_mask": None,
                "input_reference_image": "./assets/samples/portrait/human_1.jpg",
                "save_path": "examples/outputs/portrait_human_1.jpg",
                "instruction": "Maintain the facial features, A girl is wearing a neat police uniform and sporting a badge. She is smiling with a friendly and confident demeanor. The background is blurred, featuring a cartoon logo.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 10000000,
                "repainting_scale": 1.0,
                "edit_type": "repainting"
            },
            {
                "input_image": None,
                "input_mask": None,
                "input_reference_image": "./assets/samples/subject/subject_1.jpg",
                "save_path": "examples/outputs/subject_subject_1.jpg",
                "instruction": "Display the logo in a minimalist style printed in white on a matte black ceramic coffee mug, alongside a steaming cup of coffee on a cozy cafe table.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 10000000,
                "repainting_scale": 1.0,
                "edit_type": "repainting"
            },
            {
                "input_image": "./assets/samples/application/photo_editing/1_2_edit.jpg",
                "input_mask": "./assets/samples/application/photo_editing/1_2_m.webp",
                "input_reference_image": "./assets/samples/application/photo_editing/1_ref.png",
                "save_path": "examples/outputs/photo_editing_1.jpg",
                "instruction": "The item is put on the table.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 8006019,
                "repainting_scale": 1.0,
                "edit_type": "repainting"
            },
            {
                "input_image": "./assets/samples/application/logo_paste/1_1_edit.png",
                "input_mask": "./assets/samples/application/logo_paste/1_1_m.png",
                "input_reference_image": "assets/samples/application/logo_paste/1_ref.png",
                "save_path": "examples/outputs/logo_paste_1.jpg",
                "instruction": "The logo is printed on the headphones.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 934582264,
                "repainting_scale": 1.0,
                "edit_type": "repainting"
            },
            {
                "input_image": "./assets/samples/application/try_on/1_1_edit.png",
                "input_mask": "./assets/samples/application/try_on/1_1_m.png",
                "input_reference_image": "assets/samples/application/try_on/1_ref.png",
                "save_path": "examples/outputs/try_on_1.jpg",
                "instruction": "The woman dresses this skirt.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 934582264,
                "repainting_scale": 1.0,
                "edit_type": "repainting"
            },
            {
                "input_image": "./assets/samples/portrait/human_1.jpg",
                "input_mask": "assets/samples/application/movie_poster/1_2_m.webp",
                "input_reference_image": "assets/samples/application/movie_poster/1_ref.png",
                "save_path": "examples/outputs/movie_poster_1.jpg",
                "instruction": "{image}, the man faces the camera.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 3999647,
                "repainting_scale": 1.0,
                "edit_type": "repainting"
            },
            {
                "input_image": "./assets/samples/application/sr/sr_tiger.png",
                "input_mask": "./assets/samples/application/sr/sr_tiger_m.webp",
                "input_reference_image": None,
                "save_path": "examples/outputs/mario_recolorizing_1.jpg",
                "instruction": "{image} features a close-up of a young, furry tiger cub on a rock. The tiger, which appears to be quite young, has distinctive orange, "
                               "black, and white striped fur, typical of tigers. The cub's eyes have a bright and curious expression, and its ears are perked up, "
                               "indicating alertness. The cub seems to be in the act of climbing or resting on the rock. The background is a blurred grassland with trees, "
                               "but the focus is on the cub, which is vividly colored while the rest of the image is in grayscale, drawing attention to the tiger's details."
                               " The photo captures a moment in the wild, depicting the charming and tenacious nature of this young tiger,"
                               " as well as its typical interaction with the environment.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 199999,
                "repainting_scale": 0.0,
                "edit_type": "no_preprocess"
            },
            {
                "input_image": "./assets/samples/application/photo_editing/1_ref.png",
                "input_mask": "./assets/samples/application/photo_editing/1_1_orm.webp",
                "input_reference_image": None,
                "save_path": "examples/outputs/mario_repainting_1.jpg",
                "instruction": "a blue hand",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 63401,
                "repainting_scale": 1.0,
                "edit_type": "repainting"
            },
            {
                "input_image": "./assets/samples/application/photo_editing/1_ref.png",
                "input_mask": "./assets/samples/application/photo_editing/1_1_rm.webp",
                "input_reference_image": None,
                "save_path": "examples/outputs/mario_repainting_2.jpg",
                "instruction": "Mechanical  hands like a robot",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 59107,
                "repainting_scale": 1.0,
                "edit_type": "repainting"
            },
            {
                "input_image": "./assets/samples/control/1_1.webp",
                "input_mask": "./assets/samples/control/1_1_m.webp",
                "input_reference_image": None,
                "save_path": "examples/outputs/control_recolorizing.jpg",
                "instruction": "{image} Beautiful female portrait, Robot with smooth White transparent carbon shell, rococo detailing, Natural lighting, Highly detailed, Cinematic, 4K.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 9652101,
                "repainting_scale": 0.0,
                "edit_type": "recolorizing"
            },
            {
                "input_image": "./assets/samples/control/1_1.webp",
                "input_mask": "./assets/samples/control/1_1_m.webp",
                "input_reference_image": None,
                "save_path": "examples/outputs/control_depth.jpg",
                "instruction": "{image} Beautiful female portrait, Robot with smooth White transparent carbon shell, rococo detailing, Natural lighting, Highly detailed, Cinematic, 4K.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 14979476,
                "repainting_scale": 0.0,
                "edit_type": "depth_repainting"
            },
            {
                "input_image": "./assets/samples/control/1_1.webp",
                "input_mask": "./assets/samples/control/1_1_m.webp",
                "input_reference_image": None,
                "save_path": "examples/outputs/control_contour.jpg",
                "instruction": "{image} Beautiful female portrait, Robot with smooth White transparent carbon shell, rococo detailing, Natural lighting, Highly detailed, Cinematic, 4K.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 4227292472,
                "repainting_scale": 0.0,
                "edit_type": "contour_repainting"
            }
        ]