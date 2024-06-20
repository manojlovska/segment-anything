from finetuning.experiments.fine_tuning_exp import SAMExp
from torch.utils.data import DataLoader

sam_exp = SAMExp()

train_dataset = sam_exp.get_train_dataset()

# example_dataloader = DataLoader(
#     train_dataset,
#     1,
#     shuffle=True
#     )

# batch = next(iter(example_dataloader))

train_loader = sam_exp.get_train_loader(batch_size=1)

for batch in train_loader:
    # image = batch["image_info"]["image"]
    # gt_masks = batch["ground_truth_masks"]
    # bboxes = batch["bboxes"]

    # print(image)
    # print(gt_masks)
    # print(bboxes)
    # print(batch["image"])
    print(batch)


    import pdb
    pdb.set_trace()






dataloader_iterator = iter(train_loader)
for i in range(5):
    batch = next(dataloader_iterator)

    import pdb
    pdb.set_trace()