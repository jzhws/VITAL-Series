import json
import os
from tqdm import tqdm
# with open(f"scoring_dist_converted_qualified.json", 'r') as f:
#             mos_dict = json.load(f)
# qualified_mos_dict=[]
# num_low=0
# num_poor=0
# num_fair=0
# num_good=0
# num_high=0
# for file in tqdm(mos_dict):
#     # print(file)
#     if 0<=file["gt_score_norm"]<1:
#         num_low+=1
#         qualified_mos_dict.append(file)
#     if 1<=file["gt_score_norm"]<2:
#         num_poor+=1
#         qualified_mos_dict.append(file)
#     if 2<=file["gt_score_norm"]<3:
#         num_fair+=1
#         if num_good>100000:
#             continue
#     if 3<=file["gt_score_norm"]<4:
#         num_good+=1
#         if num_good>60000:
#             continue
#     if 4<=file["gt_score_norm"]<5:
#         num_high+=1
#         if num_high>30000:
#             continue
#     qualified_mos_dict.append(file)
# print(num_low,num_poor,num_fair,num_good,num_high)
with open(f"/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/rating_videos/YT_UGC.json", 'r') as f:
            mos_dict = json.load(f)
# print(len(mos_dict))
# with open(f"/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/data/picked_pairs_image.json", 'r') as f:
#             mos_dict = json.load(f)
# print(len(mos_dict))
# with open(f"/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/data/scoring_dist_converted_mix_dual.json", 'r') as f:
#             mos_dict = json.load(f)
# print(len(mos_dict))
# with open(f"/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/data/scoring_dist_converted_mix_dual.json", 'r') as f:
#             mos_dict = json.load(f)
# print(len(mos_dict))
# dual_list=[]
# # for i in range(len(mos_dict)//2):
# #     dual_list.append([mos_dict[2*i],mos_dict[2*i+1]])
# with open(f"/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/data/scoring_dist_converted_image.json", 'r') as f:
#             mos_dict1 = json.load(f)
# mos_dict.extend(mos_dict1)
# for i in range(len(mos_dict)//2):
#     dual_list.append([mos_dict[2*i],mos_dict[2*i+1]])
# qualified_mos_dict=mos_dict
# dual_list=[]
# for i in range(len(mos_dict)//2):
#     dual_list.append()
# for file in tqdm(mos_dict):
#     # print(file)
#     if file[0]["label"]!=2:

#         qualified_mos_dict.append(file)
# print(num_low,num_poor,num_fair,num_good,num_high)

# with open(f"scoring_LSVQ.json", 'r') as f:
#             mos_dict = json.load(f)
# qualified_mos_dict=[]
# num_low=0
# num_poor=0
# num_fair=0
# num_good=0
# num_high=0
# for file in tqdm(mos_dict):
#     # print(file)
#     if "low" in file["conversations"][1]["value"]:
#         num_low+=1
#     if "poor" in file["conversations"][1]["value"]:
#         num_poor+=1
#     if "fair" in file["conversations"][1]["value"]:
#         num_fair+=1
#     if "good" in file["conversations"][1]["value"]:
#         num_good+=1
#     if "high" in file["conversations"][1]["value"]:
#         num_high+=1
# print(num_low,num_poor,num_fair,num_good,num_high)
qualified_mos_dict=[]
print(len(mos_dict))
for file in mos_dict:
    for video in os.listdir("/mnt/shared-storage-user/jiaziheng/tos/jiaziheng/VQA++/YT-UGC"):
        if file["video"][:-4] in video:
            print(1)
            file["video"]=video
            qualified_mos_dict.append(file)
    # if os.path.exists('/mnt/shared-storage-user/jiaziheng/pretrain/youtube/'+file["video"]):
        # print("文件或目录存在")
        # qualified_mos_dict.append(file)
print(len(qualified_mos_dict))
with open(f'./YT_UGC1.json', "w", encoding='utf-8') as f:
    json.dump(qualified_mos_dict, f, indent=4)