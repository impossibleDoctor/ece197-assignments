import torch
import torchvision

from drinks_utils import get_drinks
from train import get_transform
import presets
import utils
from engine import evaluate

# Unfortunately I have loaded this class when I trained my model,
# So below is just a dangling class
class ArgsLike:
  def __repr__(self):
    args = vars(self)
    out = ""
    for key in args:
      out = out + f"{key}: {args[key]} \n"
    
    return out

def main():

    dataset_test = get_drinks("drinks", "test", presets.DetectionPresetEval())

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=4, collate_fn=utils.collate_fn
    )

    device = torch.device("cuda")

    num_classes = 4
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes, pretrained=False)
    model.to(device)

    model_path = "checkpoints\model_25.pth"
    checkpoint = torch.load(model_path)

    print(checkpoint.keys())

    model.load_state_dict(checkpoint['model'])

    evaluate(model, data_loader_test, device)

if __name__ == "__main__":
    main()