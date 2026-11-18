# @file types.py
# @brief 書き出しの型
# @author Kohei Yamaguchi, Claude

from typing import Literal

ExportLayout = Literal["channels_last", "channels_first", "hwc", "chw"]
