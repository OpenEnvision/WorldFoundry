from .base_options import BaseOptions, str2bool


class DemoOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument("--inference_mode", type=str, default="tracks_from_first_to_every_other_frame")
        parser.add_argument("--visualization_modes", type=str, nargs="+", default=["overlay mask"])
        # parser.add_argument("--video_path", type=str, default="orange.mp4")
        parser.add_argument("--save_tracks", type=str2bool, nargs="?", const=True, default=False)
        parser.add_argument("--recompute_tracks", type=str2bool, nargs="?", const=True, default=False)
        parser.add_argument("--overlay_factor", type=float, default=0.75)
        parser.add_argument("--rainbow_mode", type=str, default="left_right", choices=["left_right", "up_down"])
        parser.add_argument("--save_mode", type=str, default="video", choices=["image", "video"])
        parser.add_argument("--spaghetti_radius", type=float, default=1.5)
        parser.add_argument("--spaghetti_length", type=int, default=40)
        parser.add_argument("--spaghetti_grid", type=int, default=30)
        parser.add_argument("--spaghetti_scale", type=float, default=2)
        parser.add_argument("--spaghetti_every", type=int, default=10)
        parser.add_argument("--spaghetti_dropout", type=float, default=0)
        parser.set_defaults(
            data_root="datasets/demo", name="demo", batch_size=1, height=480, width=856, num_tracks=8192
        )  # ychange:240 xchange:428

        parser.add_argument("--read-prompt-file", type=str, default="../meta_data/motion_binding.json")
        parser.add_argument("--video-path", required=True, type=str)
        parser.add_argument("--mask_folder", type=str, default="../output_motion_binding_seg")
        parser.add_argument("--t2v-model", type=str, required=True)
        parser.add_argument("--output-path", default="../csv_motion_binding", type=str)
        parser.add_argument("--output_dir", "-o", type=str, default="../output_motion_binding", help="output directory")
        return parser
