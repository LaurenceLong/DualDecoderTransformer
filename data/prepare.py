import re
from typing import List


def split_sentence(sentence: str, max_box_length=50) -> List[str]:
    # 使用正则表达式分割句子
    splits = re.split(r'(?<=[.!?+\-*/=])+', sentence)
    result = []
    for split in splits:
        if len(split) > max_box_length:
            # 使用逗号、分号等进行次级分割
            subsplits = re.split(r'(?<=[,;:])+', split)
            result.extend(subsplits)
        else:
            result.append(split)
    return result


if __name__ == "__main__":
    s = "RuntimeError: The shape of the 2D attn_mask is torch.Size([50, 50]), but should be (32, 32)."
    s = " 114124 * 1123 = 212312"
    slist = split_sentence(s)
    print(slist)
