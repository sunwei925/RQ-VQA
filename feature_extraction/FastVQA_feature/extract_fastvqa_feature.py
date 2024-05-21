import argparse
import os.path

import numpy as np
import torch
import yaml
from tqdm import tqdm

import fastvqa.datasets as datasets
import fastvqa.models as models

sample_types = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]


def inference_set(inf_loader, model, device, save_path, set_name="na"):
    print(f"Validating for {set_name}.")
    keys = []
    for i, (data, name) in enumerate(tqdm(inf_loader, desc="Validating")):
        video = {}
        for key in sample_types:
            if key not in keys:
                keys.append(key)
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h,
                                                w).permute(0, 2, 1, 3, 4, 5).reshape(b * data["num_clips"][key], c,
                                                                                     t // data["num_clips"][key], h, w)
        with torch.no_grad():
            feat_no_pool = model(video, reduce_scores=False)
            feat = torch.mean(feat_no_pool, dim=0)
            feat = torch.mean(feat, dim=-1)
            feat = torch.mean(feat, dim=-1)
            feat = torch.mean(feat, dim=-1)
            feat = feat.cpu().numpy()
            save_filename = os.path.join(save_path, name[0].split("/")[-1][:-4])
            np.save(save_filename, feat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="options/fast-b_NTIRE_UGC.yml", help="the option file"
    )
    parser.add_argument(
        "-p", "--save_path", type=str, default="")

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)

    state_dict = torch.load(opt["test_load_path"], map_location=device)["state_dict"]

    if "test_load_path_aux" in opt:
        aux_state_dict = torch.load(opt["test_load_path_aux"], map_location=device)["state_dict"]

        from collections import OrderedDict

        fusion_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "fragments")
            else:
                ki = k
            fusion_state_dict[ki] = v

        for k, v in aux_state_dict.items():
            if k.startswith("frag"):
                continue
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "resize")
            else:
                ki = k
            fusion_state_dict[ki] = v

        state_dict = fusion_state_dict

    model.load_state_dict(state_dict, strict=True)

    for key in opt["data"].keys():

        if "val" not in key and "test" not in key:
            continue

        val_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )
        print(val_loader)
        print(len(val_loader))
        inference_set(
            val_loader,
            model,
            device, args.save_path,
            set_name=key,
        )


if __name__ == "__main__":
    main()
