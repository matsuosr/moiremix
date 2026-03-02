#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ImageNet Training (PixMix-ready) with Generic On-the-fly Mixing
- Supports: PixMix, LayerMix, DiffuseMix
- Online Generation: MoireDB, ColoredFractalDB via mixing_image_generators
"""

import argparse
import json
import os
import sys
import random
import shutil
import time
from datetime import datetime
import warnings
import math
import tempfile
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as tvm

# ---------------------------------------------------------
# ★ Import On-the-fly Generators
# ---------------------------------------------------------
# ディレクトリ構成:
# onthefly_root/
#   ├── train_onthefly.py
#   └── mixing_image_generators/
#       ├── __init__.py
#       ├── base.py
#       ├── moire.py
#       └── fractal.py
try:
    from mixing_image_generators import create_generator, BaseGenerator
except ImportError:
    print("[Warning] 'mixing_image_generators' package not found or incomplete.")
    print("Ensure you have the 'mixing_image_generators' directory in the same path.")
    # Fallback or exit logic could go here
    create_generator = None

# リポジトリ内の補助
import pixmix_utils as utils
from calibration_tools import *
from aug.layermix import LayerMixDataset
from aug.gridmask import GridMask
from aug.diffusemix import DiffuseMixDataset
from aug.ipmix import IPMixDataset
from aug.official_defaults_imagenet_vitb224 import get_defaults, get_extended_defaults, TIMM_VERSION
from mixing_presets import DEFAULT_PRESETS, resolve_preset

# =========================================================
# 基本設定
# =========================================================
utils.IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


def _format_candidates(candidates):
    return ", ".join([f"{p} (exists={os.path.isdir(p)})" for p in candidates])


def resolve_imagenet_split_dir(path, split):
    candidates = []
    if path is None:
        raise ValueError(f"{split} path is None")
    path = os.path.abspath(path)
    candidates.append(path)
    if os.path.isdir(path) and os.path.basename(os.path.normpath(path)) == split:
        return path, candidates
    split_path = os.path.join(path, split)
    candidates.append(split_path)
    if os.path.isdir(split_path):
        return split_path, candidates
    raise ValueError(
        f"Failed to resolve ImageNet {split} directory. "
        f"Tried: {_format_candidates(candidates)}"
    )


def _count_image_extensions(root, max_files=200000):
    counts = {}
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            counts[ext] = counts.get(ext, 0) + 1
            total += 1
            if total >= max_files:
                break
        if total >= max_files:
            break
    return counts

# ---------------------------------------------------------
# ImageNet クラス定義
# ---------------------------------------------------------
all_classes = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267', 'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02699494', 'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516', 'n02823428', 'n02823750', 'n02825657', 'n02834397', 'n02835271', 'n02837789', 'n02840245', 'n02841315', 'n02843684', 'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02871525', 'n02877765', 'n02879718', 'n02883205', 'n02892201', 'n02892767', 'n02894605', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936', 'n02917067', 'n02927161', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159', 'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003', 'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632', 'n03042490', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03160309', 'n03179701', 'n03180011', 'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941', 'n03208938', 'n03216828', 'n03218198', 'n03220513', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574', 'n03272010', 'n03272562', 'n03290653', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051', 'n03384352', 'n03388043', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03457902', 'n03459775', 'n03461385', 'n03467068', 'n03476684', 'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03529860', 'n03530642', 'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480', 'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03637318', 'n03642806', 'n03649909', 'n03657121', 'n03658185', 'n03661043', 'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03697007', 'n03706229', 'n03709823', 'n03710193', 'n03710637', 'n03710721', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733281', 'n03733805', 'n03742115', 'n03743016', 'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681', 'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863', 'n03871628', 'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397', 'n03887697', 'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03899768', 'n03902125', 'n03903868', 'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711', 'n03967562', 'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04008634', 'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434', 'n04070727', 'n04074963', 'n04081281', 'n04086273', 'n04090263', 'n04099969', 'n04111531', 'n04116512', 'n04118538', 'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04152593', 'n04153751', 'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04200800', 'n04201297', 'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04239074', 'n04243546', 'n04251144', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257', 'n04264628', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04296562', 'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04356056', 'n04357314', 'n04366367', 'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04372370', 'n04376876', 'n04380533', 'n04389033', 'n04392985', 'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04443257', 'n04447861', 'n04456115', 'n04458633', 'n04461696', 'n04462240', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04493381', 'n04501370', 'n04505470', 'n04507155', 'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04523525', 'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04592741', 'n04596742', 'n04597913', 'n04599235', 'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06359193', 'n06596364', 'n06785654', 'n06794110', 'n06874185', 'n07248320', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07590611', 'n07613480', 'n07614500', 'n07615774', 'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146', 'n07836838', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n07892512', 'n07920052', 'n07930864', 'n07932039', 'n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635', 'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141']
classes_chosen_1000 = all_classes
assert len(classes_chosen_1000) == 1000

imagenet_r_wnids = ['n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677']
imagenet_r_wnids.sort()
classes_chosen_200 = imagenet_r_wnids[:]
assert len(classes_chosen_200) == 200
imagenet_r_mask = [wnid in classes_chosen_200 for wnid in all_classes]

imagenet_a_wnids = ['n03355925', 'n03255030', 'n02504458', 'n01847000', 'n01910747', 'n02037110', 'n12144580', 'n03388043', 'n01531178', 'n02883205', 'n04131690', 'n07697313', 'n02951358', 'n02190166', 'n04456115', 'n03840681', 'n04347754', 'n04310018', 'n02980441', 'n04208210', 'n02259212', 'n01580077', 'n04086273', 'n01774750', 'n03014705', 'n02701002', 'n03888257', 'n02174001', 'n02895154', 'n04606251', 'n03721384', 'n02280649', 'n02051845', 'n03891332', 'n03384352', 'n02177972', 'n07753592', 'n02281787', 'n04235860', 'n03584829', 'n02233338', 'n04146614', 'n12057211', 'n02007558', 'n02356798', 'n03250847', 'n04033901', 'n01698640', 'n02129165', 'n07831146', 'n07714990', 'n02690373', 'n03026506', 'n02085620', 'n04141076', 'n04509417', 'n03444034', 'n02325366', 'n01694178', 'n03854065', 'n04344873', 'n03617480', 'n04389033', 'n02793495', 'n02279972', 'n03788195', 'n01843383', 'n02730930', 'n04366367', 'n04118538', 'n02317335', 'n02165456', 'n02133161', 'n04591713', 'n04019541', 'n07749582', 'n04067472', 'n01944390', 'n02879718', 'n02486410', 'n03452741', 'n01498041', 'n01784675', 'n04317175', 'n02906734', 'n02106550', 'n01677366', 'n01986214', 'n02948072', 'n02672831', 'n01820546', 'n03804744', 'n03717622', 'n02454379', 'n07718472', 'n04507155', 'n01882714', 'n03724870', 'n01833805', 'n04275548', 'n04252225', 'n02802426', 'n04133789', 'n03935335', 'n03590841', 'n01914609', 'n03124043', 'n04099969', 'n04179913', 'n02119022', 'n07697537', 'n01735189', 'n04254120', 'n02676566', 'n02127052', 'n01687978', 'n03666591', 'n01770393', 'n02814860', 'n02077923', 'n04270147', 'n02236044', 'n01819313', 'n04355338', 'n02206856', 'n01534433', 'n04376876', 'n04147183', 'n04532670', 'n02777292', 'n02445715', 'n01631663', 'n04039381', 'n04540053', 'n03837869', 'n03187595', 'n04482393', 'n02106662', 'n01924916', 'n02782093', 'n03125729', 'n02669723', 'n02655020', 'n07760859', 'n02231487', 'n02837789', 'n02009912', 'n12267677', 'n09229709', 'n02346627', 'n01558993', 'n03982430', 'n02992211', 'n02797295', 'n02361337', 'n04252077', 'n03291819', 'n01641577', 'n01669191', 'n01614925', 'n02410509', 'n02123394', 'n07583066', 'n03720891', 'n02110958', 'n01855672', 'n07734744', 'n03594945', 'n02787622', 'n02999410', 'n09472597', 'n03775071', 'n02268443', 'n04399382', 'n07768694', 'n02492035', 'n11879895', 'n03445924', 'n07695742', 'n03443371', 'n09835506', 'n03325584', 'n03670208', 'n04442312', 'n01770081', 'n02815834', 'n02226429', 'n02219486', 'n03417042', 'n03196217', 'n04554684', 'n04562935', 'n01985128', 'n02099601', 'n09246464', 'n07720875', 'n03223299', 'n01616318', 'n03483316', 'n02137549']
imagenet_a_wnids.sort()
assert len(imagenet_a_wnids) == 200
imagenet_a_mask = [wnid in imagenet_a_wnids for wnid in all_classes]

# =========================================================
# 引数
# =========================================================
parser = argparse.ArgumentParser(description='ImageNet Training (PixMix-ready)')

# データ
parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'cifar10', 'cifar100'],
                    help='Dataset type (default: imagenet)')
parser.add_argument('--data', default='', type=str,
                    help='Root directory for CIFAR datasets')
parser.add_argument('--input-size', default=None, type=int,
                    help='Input image size (default: 224 for ImageNet, 32 for CIFAR)')
parser.add_argument('--data-standard', default="data/imagenet/train/", help='Path to training dataset')
parser.add_argument('--data-val',     default="data/imagenet/val/",   help='Path to validation dataset')
parser.add_argument('--imagenet-r-dir', default="data/imagenet_r/",   help='Path to ImageNet-R')
parser.add_argument('--imagenet-c-dir', default="data/imagenet_c/",   help='Path to ImageNet-C')

# PixMix
parser.add_argument(
    '--mixing-set',
    default='',
    help=(
        "Path or preset name for mixing set. "
        "If omitted for ipmix/diffusemix/layermix, paper presets are used."
    ),
)
parser.add_argument('--mixing-set-preset', default='',
                    help='Preset name for mixing set (resolved via mixing_presets.py)')
parser.add_argument('--mixing-method', default='pixmix', choices=['pixmix', 'layermix', 'diffusemix', 'ipmix'],
                    help='Mixing strategy when using a mixing set')
parser.add_argument('--aug-severity', default=None, type=int,
                    help='Severity of base augmentation operators (official default if unset)')
parser.add_argument('--beta', default=None, type=int,
                    help='Severity of mixing (official default if unset)')
parser.add_argument(
    '--k',
    default=None,
    type=int,
    help='Mixing iterations (official default if unset)',
)
parser.add_argument('--all-ops', '-all', action='store_true',
                    help='Use all ops (+brightness/contrast/color/sharpness)')
parser.add_argument('--layermix-depth', type=int, default=None,
                    help='LayerMix depth (official default if unset)')
parser.add_argument('--layermix-width', type=int, default=None,
                    help='LayerMix width (official default if unset)')
parser.add_argument('--layermix-magnitude', type=int, default=None,
                    help='LayerMix magnitude (official default if unset)')
parser.add_argument('--layermix-blending', type=float, default=None,
                    help='LayerMix blending (official default if unset)')
parser.add_argument('--ipmix-t', type=int, default=None,
                    help='IPMix max depth (official default if unset)')
parser.add_argument('--gridmask', action='store_true', help='Enable GridMask augmentation')
parser.add_argument('--gridmask-d-min', type=int, default=None, help='GridMask d1 (official default if unset)')
parser.add_argument('--gridmask-d-max', type=int, default=None, help='GridMask d2 (official default if unset)')
parser.add_argument('--gridmask-ratio', type=float, default=None, help='GridMask ratio (official default if unset)')
parser.add_argument('--gridmask-rotate', type=float, default=None, help='GridMask max rotation (official default if unset)')
parser.add_argument('--gridmask-fill', type=float, default=0.0, help='GridMask fill value (set to -1 for black)')
parser.add_argument('--gridmask-prob', type=float, default=None, help='GridMask max probability (official default if unset)')
parser.add_argument('--gridmask-mode', type=int, default=None, help='GridMask mode (official default if unset)')
parser.add_argument('--gridmask-prob-schedule', choices=['linear', 'const'], default='linear',
                    help='GridMask probability schedule')
parser.add_argument('--gridmask-prob-warmup-frac', type=float, default=0.8,
                    help='Warmup fraction for linear schedule')
parser.add_argument('--diffusemix-fractal-set', default='',
                    help='Optional fractal set path/preset for DiffuseMix (defaults to mixing-set)')
parser.add_argument('--diffusemix-fractal-preset', default='',
                    help='Preset name for DiffuseMix fractal set (resolved via mixing_presets.py)')
parser.add_argument('--diffusemix-fractal-lambda', type=float, default=None,
                    help='DiffuseMix fractal mixing coefficient (official default if unset)')
parser.add_argument('--diffusemix-alpha', type=float, default=None,
                    help='DiffuseMix blend factor for original/generated images (official default if unset)')
parser.add_argument('--diffusemix-beta', type=float, default=None,
                    help='DiffuseMix blend factor with fractal image (official default if unset)')
parser.add_argument('--diffusemix-concat-prob', type=float, default=None,
                    help='Probability of concatenation in DiffuseMix (official default if unset)')

# ==========================================
# ★ Generic On-the-fly Mixing Args
# ==========================================
parser.add_argument('--online-mixing', action='store_true',
                    help='Enable on-the-fly mixing image generation (overrides --mixing-set)')
parser.add_argument('--online-backend', default='moire', choices=['moire', 'fractal', 'colorbackground', 'coloredmoire', 'deadleaves', 'perlin', 'stripe', 'fourier2019', 'afa'],
                    help='Backend algorithm for online generation')

# Moire Params
parser.add_argument('--online-moire', action='store_true', help='Legacy alias for --online-mixing --online-backend moire')
parser.add_argument('--online-moire-freq-min', type=int, default=1,
                    help='Minimum frequency candidate for online Moire (inclusive)')
parser.add_argument('--online-moire-freq-max', type=int, default=100,
                    help='Maximum frequency candidate for online Moire (inclusive)')
parser.add_argument('--online-moire-centers-min', type=int, default=1,
                    help='Minimum number of centers for online Moire (inclusive)')
parser.add_argument('--online-moire-centers-max', type=int, default=3,
                    help='Maximum number of centers for online Moire (inclusive)')
parser.add_argument('--online-moire-margin', type=float, default=0.08,
                    help='Relative margin from borders for Moire centers placement (0..0.5)')

# Fractal Params (Onthefly Colored FractalDB)
parser.add_argument('--online-fractal-iters', type=int, default=40000, help='IFS iterations')
parser.add_argument('--online-fractal-instances-min', type=int, default=1)
parser.add_argument('--online-fractal-instances-max', type=int, default=3)
parser.add_argument('--online-fractal-scale-min', type=float, default=0.4)
parser.add_argument('--online-fractal-scale-max', type=float, default=0.85)

# Dead Leaves Params
parser.add_argument('--online-deadleaves-variant', type=str, default='shapes',
                    choices=['shapes', 'squares', 'oriented', 'textured'],
                    help='Dead Leaves variant for on-the-fly generator')
parser.add_argument('--online-deadleaves-shapes-min', type=int, default=250,
                    help='Minimum number of shapes per Dead Leaves image')
parser.add_argument('--online-deadleaves-shapes-max', type=int, default=400,
                    help='Maximum number of shapes per Dead Leaves image')
parser.add_argument('--online-deadleaves-radius-min', type=float, default=4.0,
                    help='Minimum radius of Dead Leaves shapes')
parser.add_argument('--online-deadleaves-radius-max', type=float, default=40.0,
                    help='Maximum radius of Dead Leaves shapes')
parser.add_argument('--online-deadleaves-bg', type=str, default='uniform',
                    choices=['uniform', 'black', 'white'],
                    help='Background color style for Dead Leaves images')

# Perlin Noise Params
parser.add_argument('--online-perlin-mode', type=str, default='fbm',
                    choices=['fbm', 'perlin'],
                    help='Noise composition type for on-the-fly Perlin backend')
parser.add_argument('--online-perlin-tileable', action='store_true',
                    help='Generate tileable Perlin textures')
parser.add_argument('--online-perlin-perlin-scale', type=float, default=64.0,
                    help='Base scale when using pure Perlin mode')
parser.add_argument('--online-perlin-octaves-min', type=int, default=4,
                    help='Minimum number of octaves for fBM mode')
parser.add_argument('--online-perlin-octaves-max', type=int, default=7,
                    help='Maximum number of octaves for fBM mode')
parser.add_argument('--online-perlin-scale-min', type=float, default=32.0,
                    help='Minimum base scale for fBM')
parser.add_argument('--online-perlin-scale-max', type=float, default=96.0,
                    help='Maximum base scale for fBM')
parser.add_argument('--online-perlin-persistence-min', type=float, default=0.45,
                    help='Minimum persistence for fBM')
parser.add_argument('--online-perlin-persistence-max', type=float, default=0.6,
                    help='Maximum persistence for fBM')
parser.add_argument('--online-perlin-lacunarity-min', type=float, default=1.8,
                    help='Minimum lacunarity for fBM')
parser.add_argument('--online-perlin-lacunarity-max', type=float, default=2.2,
                    help='Maximum lacunarity for fBM')

# Stripe (single plane-wave grating) Params
parser.add_argument('--online-stripe-freq-min', type=float, default=1.0,
                    help='Minimum frequency for stripe generator')
parser.add_argument('--online-stripe-freq-max', type=float, default=100.0,
                    help='Maximum frequency for stripe generator')
parser.add_argument('--online-stripe-amp', type=float, default=0.5,
                    help='Amplitude for stripe generator')

# Fourier 2019 Basis Params
parser.add_argument('--online-fourier2019-mode', type=str, default='uniform',
                    choices=['uniform', 'radial'],
                    help='Sampling mode for Fourier 2019 basis')
parser.add_argument('--online-fourier2019-r-min', type=float, default=1.0,
                    help='Minimum radial frequency for radial mode')
parser.add_argument('--online-fourier2019-r-max', type=float, default=50.0,
                    help='Maximum radial frequency for radial mode')
parser.add_argument('--online-fourier2019-amp', type=float, default=0.5,
                    help='Amplitude for Fourier 2019 basis')

# AFA (CVPR 2024) Params (reference-aligned; legacy args kept for compatibility)
parser.add_argument('--online-afa-min-str', type=float, default=None,
                    help='AFA minimum strength offset (official default if unset)')
parser.add_argument('--online-afa-mean-str', type=float, default=None,
                    help='AFA exponential mean strength (official default if unset)')
parser.add_argument('--online-afa-freq-cut', type=int, default=None,
                    help='AFA number of frequency groups sampled (official default if unset)')
parser.add_argument('--online-afa-phase-cut', type=int, default=None,
                    help='AFA number of phases sampled per frequency (official default if unset)')
parser.add_argument('--online-afa-granularity', type=int, default=None,
                    help='AFA phase granularity (official default if unset)')
parser.add_argument('--online-afa-phase-min', type=float, default=None,
                    help='AFA phase range min (official default if unset)')
parser.add_argument('--online-afa-phase-max', type=float, default=None,
                    help='AFA phase range max (official default if unset)')
parser.add_argument('--online-afa-per-channel', action='store_true', default=None,
                    help='Use per-channel AFA strengths (official default if unset)')
parser.add_argument('--online-afa-shared', action='store_false', dest='online_afa_per_channel',
                    help='Use shared AFA strength across RGB channels')
parser.add_argument('--online-afa-lambda', type=float, default=None,
                    help='[Deprecated] Alias for online_afa_mean_str')
parser.add_argument('--online-afa-f-min', type=float, default=None,
                    help='[Deprecated] Minimum frequency for AFA (overrides group range)')
parser.add_argument('--online-afa-f-max', type=float, default=None,
                    help='[Deprecated] Maximum frequency for AFA (overrides group range)')

# モデル/学習
parser.add_argument('--num-classes', type=int, choices=[10, 100, 200, 1000], default=None)
parser.add_argument('-a','--arch', default='resnet50')
parser.add_argument('-j','--workers', default=8, type=int, help='data loading workers')
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b','--batch-size', default=256, type=int)
parser.add_argument('--batch-size-val', default=256, type=int)
parser.add_argument('--optimizer', default='sgd', choices=['sgd','adamw'])
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate (bs=256 の基準)')
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--sched', default='cosine', choices=['cosine'])
parser.add_argument('--warmup-epochs', type=int, default=5)
parser.add_argument('--mixup-alpha', type=float, default=None,
                    help='MixUp alpha (official timm default if unset)')
parser.add_argument('--cutmix-alpha', type=float, default=None,
                    help='CutMix alpha (official timm default if unset)')
parser.add_argument('--mixup-cutmix-recipe', choices=['none', 'imagenet_standard'],
                    default='none', help='Optional mixup/cutmix recipe preset')
parser.add_argument('--augmix', action='store_true',
                    help='Enable AugMix data augmentation')
parser.add_argument('--augmix-width', type=int, default=None,
                    help='AugMix augmentation width (official default if unset)')
parser.add_argument('--augmix-depth', type=int, default=None,
                    help='AugMix augmentation depth (official default if unset)')
parser.add_argument('--augmix-severity', type=int, default=None,
                    help='AugMix augmentation severity (official default if unset)')
parser.add_argument('--augmix-alpha', type=float, default=None,
                    help='AugMix Dirichlet sampling parameter (official default if unset)')
parser.add_argument('--cutout', action='store_true',
                    help='Enable Cutout data augmentation for training images.')
parser.add_argument('--cutout-size', type=int, default=None,
                    help='Cutout square size in pixels (fallback default: 48).')
parser.add_argument('--auto-augment', action='store_true',
                    help='Enable torchvision AutoAugment (ImageNet policy) for training images.')
parser.add_argument('--rand-augment', action='store_true',
                    help='Enable torchvision RandAugment for training images.')
parser.add_argument('--randaug-num-ops', type=int, default=2,
                    help='Number of augmentation transformations to apply sequentially.')
parser.add_argument('--randaug-magnitude', type=int, default=9,
                    help='Magnitude for all the transformations.')

# ログ/ckpt
parser.add_argument('--save', default='checkpoints/TEMP', type=str)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')

# その他
parser.add_argument('-p','--print-freq', default=50, type=int)
parser.add_argument('-e','--evaluate', action='store_true', help='only run evaluation (val + ImageNet-R + ImageNet-C)')
parser.add_argument('--pretrained', action='store_true', help='use torchvision/timm pretrained weights (finetune)')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--no-amp', action='store_true', help='disable AMP')
parser.add_argument('--reset-head', action='store_true', help='re-init classifier head even when pretrained & 1000-class')
parser.add_argument('--dry-run', action='store_true',
                    help='Resolve defaults and exit before model/dataloader setup')
parser.add_argument('--log-first-epoch-time', action='store_true',
                    help='Log wall-clock time for the first training epoch')
parser.add_argument('--estimate-epoch-time-iters', type=int, default=0,
                    help='Estimate epoch time from first N iterations (0 disables)')
parser.add_argument('--estimate-epoch-time-exit', action='store_true',
                    help='Exit after estimating epoch time')

# PGD evaluation
parser.add_argument('--pgd-eval', action='store_true', help='Run PGD adversarial evaluation on val set')
parser.add_argument('--pgd-eps', type=float, default=1/255, help='PGD epsilon (pixel space)')
parser.add_argument('--pgd-alpha', type=float, default=(1/255)/4, help='PGD step size (pixel space)')
parser.add_argument('--pgd-steps', type=int, default=50, help='PGD steps')
parser.add_argument('--pgd-norm', type=str, default='linf', choices=['linf'], help='PGD norm type')
parser.add_argument('--pgd-seed', type=int, default=1, help='Random seed for PGD attack')
parser.add_argument('--pgd-data-val', type=str, default='', help='Optional override for PGD eval dataset path')
parser.add_argument('--pgd-batch-size', type=int, default=128, help='Batch size for PGD evaluation')
parser.add_argument('--pgd-workers', type=int, default=32, help='Workers for PGD evaluation')
parser.add_argument('--pgd-log-dir', type=str, default='', help='Directory to store PGD results (defaults to save dir)')
parser.add_argument('--pgd-results-name', type=str, default='metrics.json', help='File name for PGD metrics JSON')
parser.add_argument('--pgd-tag', type=str, default='', help='Optional tag to include in PGD log directory')
eval_mode = parser.add_mutually_exclusive_group()
eval_mode.add_argument('--pgd-only', action='store_true',
                       help='Skip val/R/C evaluations and run PGD only (requires --evaluate)')
eval_mode.add_argument('--eval-only-imagenet-c', action='store_true',
                       help='Skip val/R and run ImageNet-C evaluation only (requires --evaluate)')

# 解析用
parser.add_argument('--save-imagenet-c', action='store_true', dest='save_imagenet_c')
parser.set_defaults(save_imagenet_c=False)

# =========================================================
# Dataset ヘルパ
# =========================================================
class ImageNetSubsetDataset(datasets.ImageFolder):
    """ImageFolder that filters to a subset of ImageNet classes without symlinks."""
    def __init__(self, root, *args, **kwargs):
        if classes_chosen is None:
            raise RuntimeError("classes_chosen is not initialized")
        self._subset_classes = set(classes_chosen)
        super().__init__(root, *args, **kwargs)

    def find_classes(self, directory):
        classes = []
        class_to_idx = {}
        for entry in sorted(os.scandir(directory), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            wnid = entry.name
            if wnid not in self._subset_classes:
                continue
            classes.append(wnid)
            class_to_idx[wnid] = len(class_to_idx)
        if not classes:
            raise FileNotFoundError(f"No ImageNet classes found in {directory}")
        return classes, class_to_idx


class ImageFolderWithClassMapping(datasets.ImageFolder):
    """
    ImageFolder that reuses a reference class_to_idx mapping so that all datasets
    (val / ImageNet-R / ImageNet-C) share identical label IDs.
    """
    def __init__(self, root, ref_class_to_idx, *args, **kwargs):
        self.ref_class_to_idx = ref_class_to_idx
        super().__init__(root, *args, **kwargs)

    def find_classes(self, directory):
        classes = []
        class_to_idx = {}
        for entry in sorted(os.scandir(directory), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            wnid = entry.name
            if wnid not in self.ref_class_to_idx:
                continue
            classes.append(wnid)
            class_to_idx[wnid] = self.ref_class_to_idx[wnid]
        if not classes:
            raise FileNotFoundError(f"No ImageNet classes found in {directory}")
        return classes, class_to_idx


class RecursiveImageFolder(datasets.VisionDataset):
    """Dataset that recursively collects images under root with case-insensitive extensions."""
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.samples = []
        self.ext_counts = {}
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                self.ext_counts[ext] = self.ext_counts.get(ext, 0) + 1
                if ext in self.IMG_EXTS:
                    self.samples.append(os.path.join(dirpath, fname))
        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in mixing set: {root}. "
                f"ext_counts={self.ext_counts} allowed_exts={sorted(self.IMG_EXTS)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


class Cutout(object):
    """Apply a single Cutout square to a PIL image."""
    def __init__(self, size: int):
        self.size = max(0, int(size))

    def __call__(self, img):
        if self.size <= 0:
            return img
        w, h = img.size
        if w == 0 or h == 0:
            return img

        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        half = self.size // 2
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        if x1 >= x2 or y1 >= y2:
            return img

        img_np = np.array(img)
        if img_np.ndim == 2:
            img_np[y1:y2, x1:x2] = 0
        else:
            img_np[y1:y2, x1:x2, ...] = 0
        return Image.fromarray(img_np)

# =========================================================
# PixMix
# =========================================================
def augment_input(image, use_all_ops, severity):
    aug_list = utils.augmentations_all if use_all_ops else utils.augmentations
    op = np.random.choice(aug_list)
    return op(image.copy(), severity)

def pixmix(orig, mixing_pic, preprocess, k, beta, use_all_ops, aug_severity):
    mixings = utils.mixings
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig, use_all_ops, aug_severity))
    else:
        mixed = tensorize(orig)
    iters = np.random.randint(k + 1)
    for _ in range(iters):
        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig, use_all_ops, aug_severity))
        else:
            aug_image_copy = tensorize(mixing_pic)
        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, beta)
        mixed = torch.clip(mixed, 0, 1)
    return normalize(mixed)

class PixMixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mixing_set, preprocess, k, beta, use_all_ops, aug_severity):
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.preprocess = preprocess
        self.k = k
        self.beta = beta
        self.use_all_ops = use_all_ops
        self.aug_severity = aug_severity
    def __getitem__(self, i):
        x, y = self.dataset[i]
        j = np.random.randint(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[j]
        return pixmix(x, mixing_pic, self.preprocess, self.k, self.beta, self.use_all_ops, self.aug_severity), y
    def __len__(self):
        return len(self.dataset)

# ==========================================
# ★ Generic On-the-fly PixMix Dataset
# ==========================================
class OnlinePixMixDataset(torch.utils.data.Dataset):
    """
    Dataset that uses a generator interface to create mixing images.
    """
    def __init__(
        self,
        dataset,
        preprocess,
        generator,
        k: int,
        beta: int,
        use_all_ops: bool,
        aug_severity: int
    ):
        self.dataset = dataset
        self.preprocess = preprocess
        self.generator = generator
        self.k = k
        self.beta = beta
        self.use_all_ops = use_all_ops
        self.aug_severity = aug_severity

    def __getitem__(self, i):
        # 1. Base Image (Standard Augmented)
        x, y = self.dataset[i]

        # 2. On-the-fly Generation
        mixing_pic = self.generator.generate()
        
        # Safety resize
        if mixing_pic.size != (utils.IMAGE_SIZE, utils.IMAGE_SIZE):
            mixing_pic = mixing_pic.resize((utils.IMAGE_SIZE, utils.IMAGE_SIZE))

        # 3. PixMix
        mixed = pixmix(
            x,
            mixing_pic,
            self.preprocess,
            self.k,
            self.beta,
            self.use_all_ops,
            self.aug_severity,
        )
        return mixed, y

    def __len__(self):
        return len(self.dataset)

# =========================================================
# 便利メータ
# =========================================================
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(m) for m in self.meters]
        print('\t'.join(entries), flush=True)
    def _get_batch_fmtstr(self, num_batches):
        nd = len(str(num_batches // 1))
        fmt = '{:' + str(nd) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# =========================================================
# 精度・補助関数
# =========================================================
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        bs = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / bs))
        return res

def save_checkpoint(state, is_best, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        best_path = os.path.join(os.path.dirname(filename), 'model_best.pth.tar')
        shutil.copyfile(filename, best_path)

def get_net_results(dataloader, net, device):
    to_np = lambda x: x.data.to('cpu').numpy()
    confidence = []
    correct = []
    num_correct = 0
    net.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data  = data.to(device, non_blocking=True)
            target= target.to(device, non_blocking=True)
            output = net(data)
            pred = output.argmax(1)
            num_correct += (pred == target).sum().item()
            confidence.extend(to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist())
            correct.extend((pred == target).to('cpu').numpy().astype(np.bool_).tolist())
    return num_correct / len(dataloader.dataset), np.array(confidence), np.array(correct)

# =========================================================
# メイン
# =========================================================
best_acc1 = 0.0
classes_chosen = None

def resolve_defaults(args):
    official = get_defaults()
    extended = get_extended_defaults()
    applied = []
    applied_ext = []
    sources = {}
    ext_sources = {}

    def _apply_if_none(attr, value, label):
        if getattr(args, attr) is None:
            setattr(args, attr, value)
            applied.append(f"{label}: {attr}={value}")

    if args.augmix:
        augmix = official["augmix"]
        sources["AugMix"] = augmix["source"]
        _apply_if_none("augmix_width", augmix["mixture_width"], "AugMix")
        _apply_if_none("augmix_depth", augmix["mixture_depth"], "AugMix")
        _apply_if_none("augmix_severity", augmix["aug_severity"], "AugMix")
        _apply_if_none("augmix_alpha", augmix["alpha"], "AugMix")

    if args.cutout:
        cutout = official["cutout"]
        sources["Cutout"] = cutout["source"]
        _apply_if_none("cutout_size", cutout["size"], "Cutout(fallback)")

    if args.gridmask:
        gridmask = official["gridmask"]
        sources["GridMask"] = gridmask["source"]
        _apply_if_none("gridmask_d_min", gridmask["d_min"], "GridMask")
        _apply_if_none("gridmask_d_max", gridmask["d_max"], "GridMask")
        _apply_if_none("gridmask_ratio", gridmask["ratio"], "GridMask")
        _apply_if_none("gridmask_rotate", gridmask["rotate"], "GridMask")
        _apply_if_none("gridmask_prob", gridmask["prob"], "GridMask")
        _apply_if_none("gridmask_mode", gridmask["mode"], "GridMask")

    if args.mixing_method == "pixmix":
        pixmix = official["pixmix"]
        sources["PixMix"] = pixmix["source"]
        _apply_if_none("k", pixmix["k"], "PixMix")
        _apply_if_none("beta", pixmix["beta"], "PixMix")
        _apply_if_none("aug_severity", pixmix["aug_severity"], "PixMix")

    if args.mixing_method == "ipmix":
        ipmix = official["ipmix"]
        sources["IPMix"] = ipmix["source"]
        _apply_if_none("k", ipmix["k"], "IPMix")
        _apply_if_none("ipmix_t", ipmix["t"], "IPMix")
        _apply_if_none("beta", ipmix["beta"], "IPMix")
        _apply_if_none("aug_severity", ipmix["aug_severity"], "IPMix")

    if args.mixing_method == "diffusemix":
        diffusemix = official["diffusemix"]
        sources["DiffuseMix"] = diffusemix["source"]
        _apply_if_none("diffusemix_fractal_lambda", diffusemix["fractal_lambda"], "DiffuseMix")
        _apply_if_none("diffusemix_alpha", diffusemix["alpha"], "DiffuseMix")
        _apply_if_none("diffusemix_beta", diffusemix["beta"], "DiffuseMix")
        _apply_if_none("diffusemix_concat_prob", diffusemix["concat_prob"], "DiffuseMix")

    if args.mixing_method == "layermix":
        layermix = official["layermix"]
        sources["LayerMix"] = layermix["source"]
        _apply_if_none("layermix_depth", layermix["depth"], "LayerMix")
        _apply_if_none("layermix_width", layermix["width"], "LayerMix")
        _apply_if_none("layermix_magnitude", layermix["magnitude"], "LayerMix")
        _apply_if_none("layermix_blending", layermix["blending"], "LayerMix")

    if args.mixup_cutmix_recipe == "imagenet_standard":
        recipe = official["mixup_cutmix_recipe"]["imagenet_standard"]
        sources["MixUp/CutMix recipe"] = official["mixup_cutmix_recipe"]["source"]
        _apply_if_none("mixup_alpha", recipe["mixup_alpha"], "MixUp/CutMix recipe")
        _apply_if_none("cutmix_alpha", recipe["cutmix_alpha"], "MixUp/CutMix recipe")

    sources["MixUp"] = official["mixup"]["source"]
    sources["CutMix"] = official["cutmix"]["source"]
    _apply_if_none("mixup_alpha", official["mixup"]["alpha"], f"MixUp (timm=={TIMM_VERSION})")
    _apply_if_none("cutmix_alpha", official["cutmix"]["alpha"], f"CutMix (timm=={TIMM_VERSION})")

    if args.online_mixing and args.online_backend == "afa":
        afa = official["afa"]
        sources["AFA"] = afa["source"]
        _apply_if_none("online_afa_min_str", afa["min_str"], "AFA")
        _apply_if_none("online_afa_mean_str", afa["mean_str"], "AFA")
        _apply_if_none("online_afa_freq_cut", afa["freq_cut"], "AFA")
        _apply_if_none("online_afa_phase_cut", afa["phase_cut"], "AFA")
        _apply_if_none("online_afa_granularity", afa["granularity"], "AFA")

        afa_ext = extended.get("afa", {})
        if afa_ext:
            ext_sources["AFA"] = afa_ext.get("source", "local extension")
            if args.online_afa_phase_min is None:
                args.online_afa_phase_min = afa_ext.get("phase_min")
                applied_ext.append(f"AFA: online_afa_phase_min={args.online_afa_phase_min}")
            if args.online_afa_phase_max is None:
                args.online_afa_phase_max = afa_ext.get("phase_max")
                applied_ext.append(f"AFA: online_afa_phase_max={args.online_afa_phase_max}")
            if args.online_afa_per_channel is None:
                args.online_afa_per_channel = afa_ext.get("per_channel")
                applied_ext.append(f"AFA: online_afa_per_channel={args.online_afa_per_channel}")

    if applied:
        sources_line = "; ".join([f"{k}: {v}" for k, v in sources.items()])
        print("Using OFFICIAL defaults:", "; ".join(applied))
        print("Official default sources:", sources_line)
    if applied_ext:
        ext_line = "; ".join([f"{k}: {v}" for k, v in ext_sources.items()])
        print("Using EXTENDED defaults:", "; ".join(applied_ext))
        print("Extended default sources:", ext_line)


def collect_afa_cli_overrides(argv):
    overrides = set()
    flag_map = {
        "online_afa_min_str": "--online-afa-min-str",
        "online_afa_mean_str": "--online-afa-mean-str",
        "online_afa_freq_cut": "--online-afa-freq-cut",
        "online_afa_phase_cut": "--online-afa-phase-cut",
        "online_afa_granularity": "--online-afa-granularity",
        "online_afa_phase_min": "--online-afa-phase-min",
        "online_afa_phase_max": "--online-afa-phase-max",
    }
    for key, flag in flag_map.items():
        if flag in argv:
            overrides.add(key)
    if "--online-afa-per-channel" in argv or "--online-afa-shared" in argv:
        overrides.add("online_afa_per_channel")
    return overrides


def log_resolved_afa(args, overrides):
    resolved = (
        "Resolved AFA: "
        f"online_afa_min_str={args.online_afa_min_str}; "
        f"online_afa_mean_str={args.online_afa_mean_str}; "
        f"online_afa_freq_cut={args.online_afa_freq_cut}; "
        f"online_afa_phase_cut={args.online_afa_phase_cut}; "
        f"online_afa_granularity={args.online_afa_granularity}; "
        f"online_afa_phase_min={args.online_afa_phase_min}; "
        f"online_afa_phase_max={args.online_afa_phase_max}; "
        f"online_afa_per_channel={args.online_afa_per_channel}"
    )
    print(resolved)
    if overrides:
        override_items = [f"{key}={getattr(args, key)}" for key in sorted(overrides)]
        print("CLI overrides:", "; ".join(override_items))

def main():
    global classes_chosen, best_acc1
    afa_cli_overrides = collect_afa_cli_overrides(sys.argv)
    args = parser.parse_args()
    setattr(argparse, '_parsed_args', args)

    mixing_set_value = None
    if args.mixing_set:
        mixing_set_value = args.mixing_set
    elif args.mixing_set_preset:
        mixing_set_value = args.mixing_set_preset
    elif args.mixing_method in DEFAULT_PRESETS:
        mixing_set_value = DEFAULT_PRESETS[args.mixing_method]

    if mixing_set_value:
        args.mixing_set = resolve_preset(mixing_set_value, require_exists=True)

    if args.mixing_method == "diffusemix":
        fractal_value = None
        if not args.diffusemix_fractal_set:
            if args.diffusemix_fractal_preset:
                fractal_value = args.diffusemix_fractal_preset
            else:
                fractal_value = args.mixing_set
        else:
            fractal_value = args.diffusemix_fractal_set

        if fractal_value:
            args.diffusemix_fractal_set = resolve_preset(fractal_value, require_exists=True)

    resolve_defaults(args)
    if args.online_mixing and args.online_backend == "afa":
        log_resolved_afa(args, afa_cli_overrides)

    if args.input_size is None:
        args.input_size = 224 if args.dataset == "imagenet" else 32
    utils.IMAGE_SIZE = args.input_size

    if args.num_classes is None:
        if args.dataset == "cifar10":
            args.num_classes = 10
        elif args.dataset == "cifar100":
            args.num_classes = 100
        else:
            args.num_classes = 1000

    if args.mixing_method in ("ipmix", "diffusemix", "layermix"):
        if args.mixing_method == "ipmix":
            paper_line = (
                f"Using official defaults for ipmix: k={args.k} t={args.ipmix_t} "
                f"beta={args.beta} all_ops={args.all_ops} mixing_set={args.mixing_set}"
            )
        elif args.mixing_method == "diffusemix":
            paper_line = (
                "Using official defaults for diffusemix: "
                f"fractal_lambda={args.diffusemix_fractal_lambda} "
                f"mixing_set={args.mixing_set} fractal_set={args.diffusemix_fractal_set}"
            )
        else:
            paper_line = (
                "Using official defaults for layermix: "
                f"depth={args.layermix_depth} width={args.layermix_width} "
                f"magnitude={args.layermix_magnitude} blending={args.layermix_blending} "
                f"mixing_set={args.mixing_set}"
            )
        print(paper_line)

    print(args)

    if args.dry_run:
        print("[dry-run] Defaults resolved; exiting before model/dataloader setup.")
        return

    if args.pgd_only:
        args.pgd_eval = True

    # クラス集合を決定
    if args.dataset == "imagenet":
        if args.num_classes == 200:
            classes_chosen = classes_chosen_200
        elif args.num_classes == 1000:
            classes_chosen = classes_chosen_1000
        else:
            raise NotImplementedError

    # 再現性
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('CUDNN deterministic 有効: 速度低下の可能性あり')

    # 出力ディレクトリ
    if os.path.exists(args.save):
        if not os.path.isdir(args.save):
            raise Exception(f'{args.save} is not a dir')
    else:
        os.makedirs(args.save, exist_ok=True)
        print("Made save directory", args.save)

    # GPU/AMP
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    use_amp = (not args.no_amp) and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # モデル
    num_classes = int(args.num_classes)
    print(f"=> creating model '{args.arch}' (num_classes={num_classes}, pretrained={args.pretrained})")
    model = build_model(args.arch, args.pretrained, num_classes)
    model = model.to(device)
    # --- load checkpoint (works also with --evaluate) ---
    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        from collections import OrderedDict
        ckpt = torch.load(args.resume, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        sd = OrderedDict((k[7:], v) if k.startswith('module.') else (k, v) for k, v in sd.items())
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"=> loaded. epoch={ckpt.get('epoch')}, best_top1={ckpt.get('best_top1')}, missing={len(missing)}, unexpected={len(unexpected)}")
    # --- end load checkpoint ---


    # クリテリオン
    criterion = nn.CrossEntropyLoss().to(device)

    # 評価系 Loader（val のクラスマッピングを基準に固定）
    if args.dataset == "imagenet":
        normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        eval_resize = int(round(args.input_size * 256 / 224))
        eval_transform = transforms.Compose([
            transforms.Resize(eval_resize),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            normalize,
        ])
        val_dir, val_candidates = resolve_imagenet_split_dir(args.data_val, "val")
        if val_dir != args.data_val:
            print(f"[INFO] Resolved val dir: {val_dir} (from {args.data_val})")
            print(f"[INFO] val candidates: {_format_candidates(val_candidates)}")
        args.data_val = val_dir
        val_dataset = datasets.ImageFolder(args.data_val, transform=eval_transform)
        if len(val_dataset) == 0:
            ext_counts = _count_image_extensions(args.data_val)
            raise ValueError(
                f"val dataset is empty after resolution. dir={args.data_val} "
                f"candidates={_format_candidates(val_candidates)} "
                f"ext_counts={ext_counts} "
                f"allowed_exts={getattr(val_dataset, 'extensions', None)}"
            )
        args.ref_class_to_idx = val_dataset.class_to_idx
        args.ref_classes = val_dataset.classes
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size_val, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

        val_loader_imagenet_r = None
        if os.path.isdir(args.imagenet_r_dir):
            try:
                ds_r = ImageFolderWithClassMapping(
                    args.imagenet_r_dir, args.ref_class_to_idx,
                    transform=eval_transform
                )
                val_loader_imagenet_r = torch.utils.data.DataLoader(
                    ds_r,
                    batch_size=args.batch_size_val, shuffle=False,
                    num_workers=args.workers, pin_memory=True
                )
            except Exception as e:
                print(f"[WARN] Failed to build ImageNet-R loader: {e}")
                val_loader_imagenet_r = None
    else:
        if not args.data:
            raise ValueError("--data is required for CIFAR datasets")
        if args.dataset == "cifar10":
            normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        else:
            normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        eval_steps = []
        if args.input_size != 32:
            eval_steps.extend([transforms.Resize(args.input_size), transforms.CenterCrop(args.input_size)])
        eval_steps.extend([transforms.ToTensor(), normalize])
        eval_transform = transforms.Compose(eval_steps)
        if args.dataset == "cifar10":
            val_dataset = datasets.CIFAR10(args.data, train=False, transform=eval_transform, download=False)
        else:
            val_dataset = datasets.CIFAR100(args.data, train=False, transform=eval_transform, download=False)
        args.ref_classes = getattr(val_dataset, "classes", None)
        if args.ref_classes:
            args.ref_class_to_idx = {name: i for i, name in enumerate(args.ref_classes)}
        else:
            args.ref_class_to_idx = None
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size_val, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        val_loader_imagenet_r = None

    # 評価のみモード
    if args.evaluate:
        if args.eval_only_imagenet_c:
            if args.dataset == "imagenet":
                evaluate_c(model, normalize, device, args, ref_class_to_idx=args.ref_class_to_idx)  # ImageNet-C
            else:
                print("Skipping ImageNet-C evaluation (non-ImageNet dataset).")
        elif not args.pgd_only:
            val_loss, val_top1, val_top5 = validate(val_loader, model, criterion, device, args)
            r_top1 = r_top5 = None
            if val_loader_imagenet_r is not None:
                _, r_top1, r_top5 = validate(val_loader_imagenet_r, model, criterion, device, args, r=True)
            if args.dataset == "imagenet":
                evaluate_c(model, normalize, device, args, ref_class_to_idx=args.ref_class_to_idx)  # ImageNet-C
            with open(os.path.join(args.save, "eval_results.csv"), 'w') as f:
                f.write('val_top1,val_top5,r_top1,r_top5\n')
                f.write('%0.5f,%0.5f,%s,%s\n' % (
                    float(val_top1), float(val_top5),
                    ('%0.5f' % r_top1) if r_top1 is not None else 'NA',
                    ('%0.5f' % r_top5) if r_top5 is not None else 'NA'
                ))
        else:
            print("Skipping standard evaluations (PGD only mode).")

        if args.pgd_eval:
            evaluate_pgd(model, criterion, device, args, eval_transform, val_dataset)

        print('FINISHED EVALUATION')
        return

    # --- ここから学習専用の設定 ---
    # ViT 系なら AdamW / weight decay などを薄く上書き
    if args.arch in ('vit_tiny', 'vit_base', 'vit_b_16', 'vit_b16', 'vit_b_16_tv'):
        if args.optimizer == 'sgd':
            args.optimizer = 'adamw'
        if abs(args.lr - 0.1) < 1e-12:
            args.lr = 3e-3
        if abs(args.weight_decay - 1e-4) < 1e-12:
            args.weight_decay = 0.05
        print(f"[ViT overrides] optimizer={args.optimizer} lr={args.lr} weight_decay={args.weight_decay}")

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov=True
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.optimizer)

    # cosine + warmup（iteration 単位）
    def cosine_mult(step, total_steps):
        return 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
    scheduler_conf = dict(total_steps=None, warmup_steps=None)  # 後で埋める

    # 学習データ（PixMixの有無で分岐）
    tensorize = transforms.ToTensor()
    if args.gridmask_fill < 0:
        gridmask_fill = None
    else:
        gridmask_fill = args.gridmask_fill
    gridmask_transform = None
    if args.gridmask:
        gridmask_transform = GridMask(
            d_min=args.gridmask_d_min,
            d_max=args.gridmask_d_max,
            ratio=args.gridmask_ratio,
            rotate=args.gridmask_rotate,
            fill=gridmask_fill,
            prob=args.gridmask_prob,
            mode=args.gridmask_mode,
        )

    train_base_transforms = []
    if args.dataset == "imagenet":
        train_dir, train_candidates = resolve_imagenet_split_dir(args.data_standard, "train")
        if train_dir != args.data_standard:
            print(f"[INFO] Resolved train dir: {train_dir} (from {args.data_standard})")
            print(f"[INFO] train candidates: {_format_candidates(train_candidates)}")
        args.data_standard = train_dir
        print(f"Using training dataset at: {args.data_standard}")
        train_base_transforms.extend([
            transforms.RandomResizedCrop(
                args.input_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        if not args.data:
            raise ValueError("--data is required for CIFAR datasets")
        if args.input_size != 32:
            train_base_transforms.append(transforms.Resize(args.input_size))
        train_base_transforms.extend([
            transforms.RandomCrop(args.input_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    if args.auto_augment:
        train_base_transforms.append(
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)
        )
    if getattr(args, 'rand_augment', False):
            train_base_transforms.append(
                transforms.RandAugment(num_ops=args.randaug_num_ops, magnitude=args.randaug_magnitude)
            )
    if args.augmix:
        train_base_transforms.append(
            transforms.AugMix(
                severity=args.augmix_severity,
                mixture_width=args.augmix_width,
                chain_depth=args.augmix_depth,
                alpha=args.augmix_alpha,
            )
        )
    if gridmask_transform is not None:
        train_base_transforms.append(gridmask_transform)
    if args.cutout:
        train_base_transforms.append(Cutout(args.cutout_size))
    train_base_pil_transforms = transforms.Compose(train_base_transforms)
    if args.dataset == "imagenet":
        train_base = ImageNetSubsetDataset(
            args.data_standard,
            transform=train_base_pil_transforms
        )
    elif args.dataset == "cifar10":
        train_base = datasets.CIFAR10(args.data, train=True, transform=train_base_pil_transforms, download=False)
    else:
        train_base = datasets.CIFAR100(args.data, train=True, transform=train_base_pil_transforms, download=False)

    # ---------------------------------------------------------
    # ここから train_dataset の決定ロジック
    # ---------------------------------------------------------

    if args.mixing_method in ('layermix', 'diffusemix', 'ipmix') and not args.mixing_set:
        raise ValueError(f"{args.mixing_method.capitalize()} requires --mixing-set to be specified")

    mixing_resize = int(round(args.input_size * 256 / 224)) if args.dataset == "imagenet" else args.input_size

    # ★ On-the-fly Mixing Logic
    if args.online_mixing or args.online_moire:
        if args.mixing_method != 'pixmix':
            raise ValueError("--online-mixing supports only 'pixmix' mixing method.")
        
        # Determine backend
        backend = args.online_backend
        if args.online_moire and not args.online_mixing:
            backend = 'moire'  # backward compatibility
        
        # mixing-set が指定されていても無視することを警告
        if args.mixing_set:
            print("[INFO] --online-mixing is enabled. Arguments in --mixing-set are IGNORED.")

        print(f"Using On-the-fly [{backend}] PixMix.")
        
        # Create Generator using factory
        generator = create_generator(backend, size=utils.IMAGE_SIZE, **vars(args))

        train_dataset = OnlinePixMixDataset(
            train_base,  # ← Standard Augmentation 済み
            {'normalize': normalize, 'tensorize': tensorize},
            generator=generator,
            k=args.k,
            beta=args.beta,
            use_all_ops=args.all_ops,
            aug_severity=args.aug_severity
        )
        
        # ログ用タグ作成
        aug_tags = [f"PixMix(Online:{backend})"]
        if args.gridmask: aug_tags.append("GridMask")
        if args.augmix:   aug_tags.append("AugMix")
        print(f"{' + '.join(aug_tags)} enabled. train_size={len(train_base)}")

    # 既存の PixMix / LayerMix / DiffuseMix (フォルダ読み込み)
    elif args.mixing_set:
        mixing_set = RecursiveImageFolder(
            args.mixing_set,
            transform=transforms.Compose([
                transforms.Resize(mixing_resize),
                transforms.RandomCrop(args.input_size),
            ])
        )
        if args.mixing_method == 'layermix':
            train_dataset = LayerMixDataset(
                train_base,
                mixing_set,
                {'normalize': normalize, 'tensorize': tensorize},
                depth=args.layermix_depth,
                width=args.layermix_width,
                magnitude=args.layermix_magnitude,
                blending=args.layermix_blending,
                use_all_ops=args.all_ops,
            )
            print(
                f"len(train_base)={len(train_base)} "
                f"len(mixing_set)={len(mixing_set)} "
                f"len(train_dataset)={len(train_dataset)}"
            )
            print(f"LayerMix enabled. train_size={len(train_base)}")

        elif args.mixing_method == 'diffusemix':
            fractal_dir = args.diffusemix_fractal_set or args.mixing_set
            fractal_set = RecursiveImageFolder(
                fractal_dir,
                transform=transforms.Compose([
                    transforms.Resize(mixing_resize),
                    transforms.RandomCrop(args.input_size),
                ])
            )
            train_dataset = DiffuseMixDataset(
                train_base,
                mixing_set,
                {'normalize': normalize, 'tensorize': tensorize},
                fractal_set=fractal_set,
                alpha=args.diffusemix_alpha,
                beta=args.diffusemix_beta,
                concat_prob=args.diffusemix_concat_prob,
                fractal_lambda=args.diffusemix_fractal_lambda,
            )
            print(
                f"len(train_base)={len(train_base)} "
                f"len(mixing_set)={len(mixing_set)} "
                f"len(fractal_set)={len(fractal_set)} "
                f"len(train_dataset)={len(train_dataset)}"
            )
            print(f"DiffuseMix enabled. train_size={len(train_base)}")

        elif args.mixing_method == 'ipmix':
            train_dataset = IPMixDataset(
                train_base,
                mixing_set,
                preprocess={'normalize': normalize, 'tensorize': tensorize},
                k=args.k,
                t=args.ipmix_t,
                beta=args.beta,
                aug_severity=args.aug_severity,
                all_ops=args.all_ops,
            )
            print(
                f"len(train_base)={len(train_base)} "
                f"len(mixing_set)={len(mixing_set)} "
                f"len(train_dataset)={len(train_dataset)}"
            )
            print(f"IPMix enabled. train_size={len(train_base)}")

        elif args.mixing_method == 'pixmix':
            train_dataset = PixMixDataset(
                train_base,
                mixing_set,
                {'normalize': normalize, 'tensorize': tensorize},
                k=args.k, beta=args.beta, use_all_ops=args.all_ops, aug_severity=args.aug_severity
            )
            print(
                f"len(train_base)={len(train_base)} "
                f"len(mixing_set)={len(mixing_set)} "
                f"len(train_dataset)={len(train_dataset)}"
            )
            print(f"PixMix enabled. train_size={len(train_base)}")
        else:
            raise ValueError(f"Unsupported mixing method: {args.mixing_method}")
            
    # PixMix等を使わない通常学習
    else:
        # Standard Transforms Logic (AugMix, GridMask applied to train_base already?)
        # Note: train_base is already defined with standard transforms + augmix + gridmask.
        # So we can reuse it directly or re-wrap it if needed.
        # Original code re-wrapped it, but reusing train_base is safer and cleaner here.
        train_base.transform = transforms.Compose(train_base_transforms + [tensorize, normalize])
        train_dataset = train_base
        print(f"Standard training. train_size={len(train_dataset)}")

    train_len = len(train_dataset)
    if train_len == 0:
        if args.dataset == "imagenet":
            ext_counts = _count_image_extensions(args.data_standard)
            raise ValueError(
                f"train dataset is empty after resolution. dir={args.data_standard} "
                f"candidates={_format_candidates(train_candidates)} "
                f"ext_counts={ext_counts} "
                f"allowed_exts={getattr(train_dataset, 'extensions', None)}"
            )
        raise ValueError(f"train dataset is empty. dataset={args.dataset} data={args.data}")

    # DataLoader
    def wif(_):
        uint64_seed = torch.initial_seed()
        ss = np.random.SeedSequence([uint64_seed])
        np.random.seed(ss.generate_state(4))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, worker_init_fn=wif
    )

    # iteration 数が決まったので scheduler の係数を構築
    iters_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * iters_per_epoch
    warmup_steps = args.warmup_epochs * iters_per_epoch
    scheduler_conf.update(total_steps=total_steps, warmup_steps=warmup_steps)

    def lr_mult(step):
        if step < warmup_steps and warmup_steps > 0:
            return float(step + 1) / float(warmup_steps)
        return cosine_mult(step - warmup_steps, max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)

    # 再開
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        ckpt = torch.load(args.resume, map_location='cpu')
        start_epoch = ckpt.get('epoch', 0)
        best_acc1 = float(ckpt.get('best_top1', ckpt.get('best_acc1', 0.0)))
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt and isinstance(ckpt['scaler'], dict):
            try:
                scaler.load_state_dict(ckpt['scaler'])
            except Exception:
                pass
        print(f"=> loaded (epoch={start_epoch}, best_top1={best_acc1:.5f})")
        # scheduler を epoch/iter に同期
        consumed_steps = start_epoch * iters_per_epoch
        for _ in range(consumed_steps):
            scheduler.step()

    cudnn.benchmark = True

    # ログヘッダ
    train_log = os.path.join(args.save, 'training_log.csv')
    if not (args.resume and os.path.isfile(train_log)):
        with open(train_log, 'w') as f:
            f.write('epoch,train_loss,train_acc1,train_acc5,val_loss,val_acc1,val_acc5,R_loss,R_acc1,R_acc5\n')

    # ============================
    # メイントレーニングループ
    # ============================
    global_step = start_epoch * iters_per_epoch
    first_epoch_logged = False
    estimate_logged = False
    for epoch in range(start_epoch, args.epochs):
        if gridmask_transform is not None:
            if args.gridmask_prob_schedule == 'linear':
                warmup_epochs = max(1.0, args.epochs * args.gridmask_prob_warmup_frac)
                gridmask_transform.set_prob(epoch + 1, warmup_epochs)
            else:
                gridmask_transform.prob = gridmask_transform.st_prob

        should_estimate = (
            args.estimate_epoch_time_iters > 0
            and (not estimate_logged)
            and epoch == start_epoch
        )
        should_log_first_epoch = (
            args.log_first_epoch_time
            and (not first_epoch_logged)
            and epoch == start_epoch
            and not (args.estimate_epoch_time_exit and args.estimate_epoch_time_iters > 0)
        )

        if should_log_first_epoch:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            tr_loss, tr_top1, tr_top5, global_step, estimate_info = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                epoch,
                global_step,
                args,
                estimate_iters=args.estimate_epoch_time_iters if should_estimate else 0,
                estimate_exit=args.estimate_epoch_time_exit if should_estimate else False,
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            first_epoch_logged = True
            is_online_mixing = bool(args.online_mixing or args.online_moire)
            online_backend = args.online_backend if args.online_mixing else ("moire" if args.online_moire else "")
            payload = {
                "epoch_index": int(epoch),
                "epoch_number": int(epoch + 1),
                "train_seconds": float(elapsed),
                "iters_per_epoch": int(iters_per_epoch),
                "batch_size": int(args.batch_size),
                "workers": int(args.workers),
                "arch": str(args.arch),
                "mixing_method": str(args.mixing_method),
                "online_mixing": bool(is_online_mixing),
                "online_backend": str(online_backend),
                "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            }
            os.makedirs(args.save, exist_ok=True)
            first_epoch_json = os.path.join(args.save, "first_epoch_time.json")
            with open(first_epoch_json, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f"[TIME] first_epoch_train_seconds={elapsed:.2f} saved={first_epoch_json}")
        else:
            tr_loss, tr_top1, tr_top5, global_step, estimate_info = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                epoch,
                global_step,
                args,
                estimate_iters=args.estimate_epoch_time_iters if should_estimate else 0,
                estimate_exit=args.estimate_epoch_time_exit if should_estimate else False,
            )

        if should_estimate and estimate_info:
            estimate_logged = True
            measured_iters = int(estimate_info["measured_iters"])
            elapsed_seconds = float(estimate_info["elapsed_seconds"])
            sec_per_iter = elapsed_seconds / max(1, measured_iters)
            est_epoch_seconds = sec_per_iter * float(iters_per_epoch)
            online_backend = args.online_backend if args.online_mixing else ("moire" if args.online_moire else "")
            payload = {
                "epoch_index": int(epoch),
                "measured_iters": int(measured_iters),
                "elapsed_seconds": float(elapsed_seconds),
                "sec_per_iter": float(sec_per_iter),
                "iters_per_epoch": int(iters_per_epoch),
                "est_epoch_seconds": float(est_epoch_seconds),
                "batch_size": int(args.batch_size),
                "workers": int(args.workers),
                "arch": str(args.arch),
                "dataset": str(args.dataset),
                "mixing_method": str(args.mixing_method),
                "online_backend": str(online_backend),
                "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            }
            os.makedirs(args.save, exist_ok=True)
            estimate_json = os.path.join(args.save, "epoch_time_estimate.json")
            with open(estimate_json, 'w') as f:
                json.dump(payload, f, indent=2)
            print(
                "[EST] "
                f"iters={measured_iters} elapsed={elapsed_seconds:.2f} "
                f"sec_per_iter={sec_per_iter:.4f} est_epoch_seconds={est_epoch_seconds:.1f} "
                f"save={estimate_json}"
            )
            if args.estimate_epoch_time_exit:
                return

        print("Evaluating on validation set")
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion, device, args)
        r_loss = r_top1 = r_top5 = 0.0
        if val_loader_imagenet_r is not None:
            r_loss, r_top1, r_top5 = validate(val_loader_imagenet_r, model, criterion, device, args, r=True)

        # ログ保存
        with open(train_log, 'a') as f:
            f.write('%03d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (
                epoch + 1, tr_loss, tr_top1, tr_top5, val_loss, val_top1, val_top5, r_loss, r_top1, r_top5
            ))

        # ベスト保存
        is_best = (val_top1/100.0) > best_acc1
        best_acc1 = max(best_acc1, val_top1/100.0)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_top1': best_acc1,          # 0..1
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if use_amp else {},
        }, is_best, os.path.join(args.save, "model.pth.tar"))

    # 仕上げに ImageNet-C
    if args.dataset == "imagenet":
        evaluate_c(model, normalize, device, args, ref_class_to_idx=getattr(args, 'ref_class_to_idx', None))
    if args.pgd_eval:
        evaluate_pgd(model, criterion, device, args, eval_transform, val_dataset)
    print("FINISHED TRAINING")

# =========================================================
# 1epoch 学習
# =========================================================
def rand_bbox(tensor_size, lam):
    """Sample CutMix bounding box coordinates."""
    W = tensor_size[-1]
    H = tensor_size[-2]
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = int(np.clip(cx - cut_w // 2, 0, W))
    x2 = int(np.clip(cx + cut_w // 2, 0, W))
    y1 = int(np.clip(cy - cut_h // 2, 0, H))
    y2 = int(np.clip(cy + cut_h // 2, 0, H))
    return x1, x2, y1, y2

def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    global_step,
    args,
    estimate_iters: int = 0,
    estimate_exit: bool = False,
):
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1   = AverageMeter('Acc@1', ':6.2f')
    top5   = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [losses, top1, top5], prefix=f"Epoch: [{epoch+1}] ")

    end = time.time()
    use_mixup = args.mixup_alpha is not None and args.mixup_alpha > 0.0
    use_cutmix = args.cutmix_alpha is not None and args.cutmix_alpha > 0.0
    estimate_start = None
    estimate_elapsed = None
    estimate_measured = 0
    last_iter = -1

    for i, (images, target) in enumerate(train_loader):
        last_iter = i
        if estimate_iters > 0 and estimate_start is None:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            estimate_start = time.time()

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        mixed_images = images
        targets_a = target
        targets_b = target
        lam = 1.0

        if use_mixup or use_cutmix:
            if use_mixup and use_cutmix:
                mix_mode = 'cutmix' if np.random.rand() < 0.5 else 'mixup'
            elif use_mixup:
                mix_mode = 'mixup'
            else:
                mix_mode = 'cutmix'

            if mix_mode == 'mixup':
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                lam = max(lam, 1.0 - lam)  # enforce lambda >= 0.5 for stability
                index = torch.randperm(images.size(0), device=device)
                mixed_images = lam * images + (1.0 - lam) * images[index, :]
                targets_a = target
                targets_b = target[index]
            else:  # CutMix
                lam = np.random.beta(args.cutmix_alpha, args.cutmix_alpha)
                index = torch.randperm(images.size(0), device=device)
                mixed_images = images.clone()
                x1, x2, y1, y2 = rand_bbox(images.size(), lam)
                mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
                # adjust lam to exactly match pixel ratio
                box_area = (x2 - x1) * (y2 - y1)
                total_area = images.size(-1) * images.size(-2)
                lam = 1.0 - float(box_area) / float(total_area)
                targets_a = target
                targets_b = target[index]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            output = model(mixed_images)
            if lam != 1.0:
                loss = lam * criterion(output, targets_a) + (1.0 - lam) * criterion(output, targets_b)
            else:
                loss = criterion(output, target)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        acc1, acc5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        if i % args.print_freq == 0:
            progress.display(i)

        global_step += 1
        if estimate_iters > 0 and estimate_elapsed is None and (i + 1) >= estimate_iters:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            estimate_elapsed = time.time() - estimate_start
            estimate_measured = i + 1
            if estimate_exit:
                break

    if estimate_iters > 0 and estimate_elapsed is None and estimate_start is not None and last_iter >= 0:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        estimate_elapsed = time.time() - estimate_start
        estimate_measured = last_iter + 1

    estimate_info = None
    if estimate_iters > 0 and estimate_elapsed is not None and estimate_measured > 0:
        estimate_info = {
            "measured_iters": int(estimate_measured),
            "elapsed_seconds": float(estimate_elapsed),
        }

    return losses.avg, top1.avg, top5.avg, global_step, estimate_info

# =========================================================
# 検証
# =========================================================
@torch.no_grad()
def validate(val_loader, model, criterion, device, args, r: bool=False):
    suffix = "(ImageNet-R)" if r else "(val)"
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1   = AverageMeter('Acc@1', ':6.2f')
    top5   = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [losses, top1, top5], prefix=f"Test {suffix}: ")

    for i, (images, target) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        if i % args.print_freq == 0:
            progress.display(i)

    print(f" * {suffix} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")
    return losses.avg, top1.avg, top5.avg

# =========================================================
# ImageNet-C 簡易評価
# =========================================================
@torch.no_grad()
def evaluate_c(model, normalize, device, args, ref_class_to_idx=None):
    root = args.imagenet_c_dir
    if not os.path.isdir(root):
        print("[ImageNet-C] dir not found -> skip")
        return

    print("[ImageNet-C] evaluating ...")
    corruptions = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    results = []
    for corr in corruptions:
        corr_top1 = []
        corr_top5 = []
        for severity in range(1, 6):
            sev_root = os.path.join(root, corr, str(severity))
            if not os.path.isdir(sev_root):
                continue
            if ref_class_to_idx is not None:
                ds = ImageFolderWithClassMapping(
                    sev_root,
                    ref_class_to_idx,
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
            else:
                ds = datasets.ImageFolder(
                    sev_root,
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size_val, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )
            model.eval()
            correct1 = correct5 = total = 0
            for images, target in loader:
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                logits = model(images)
                _, pred = logits.topk(5, 1, True, True)
                total += target.size(0)
                correct1 += (pred[:, :1].squeeze(1) == target).sum().item()
                correct5 += (pred == target.view(-1, 1)).sum().item()
            top1 = 100.0 * correct1 / max(1, total)
            top5 = 100.0 * correct5 / max(1, total)
            corr_top1.append(top1)
            corr_top5.append(top5)
            print(f"    - {corr} sev{severity}: top1={top1:.2f} top5={top5:.2f}")
        if corr_top1:
            mean_top1 = sum(corr_top1) / len(corr_top1)
            mean_top5 = sum(corr_top5) / len(corr_top5)
            results.append((corr, mean_top1, mean_top5))
            print(f"  -> {corr:15s} mean top1={mean_top1:.2f} mean top5={mean_top5:.2f}")

    if args.save_imagenet_c:
        out_csv = os.path.join(args.save, "imagenet_c_results.csv")
        with open(out_csv, 'w') as f:
            f.write("corruption,top1,top5\n")
            for corr, t1, t5 in results:
                f.write(f"{corr},{t1:.4f},{t5:.4f}\n")
        print(f"[ImageNet-C] saved: {out_csv}")

# =========================================================
# ImageNet-PGD 評価
# =========================================================
def evaluate_pgd(model, criterion, device, args, eval_transform, base_dataset=None):
    if not args.pgd_eval:
        return
    if args.pgd_norm != 'linf':
        raise ValueError("Only L_inf PGD is supported currently.")

    data_root = args.pgd_data_val if args.pgd_data_val else args.data_val
    if data_root == args.data_val and base_dataset is not None:
        dataset = base_dataset
    else:
        if getattr(args, 'ref_class_to_idx', None) is not None:
            dataset = ImageFolderWithClassMapping(data_root, args.ref_class_to_idx, transform=eval_transform)
        else:
            dataset = datasets.ImageFolder(data_root, transform=eval_transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.pgd_batch_size,
        shuffle=False,
        num_workers=args.pgd_workers,
        pin_memory=True
    )

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    eps = torch.tensor([args.pgd_eps / s for s in IMAGENET_STD], device=device).view(1, 3, 1, 1)
    alpha = torch.tensor([args.pgd_alpha / s for s in IMAGENET_STD], device=device).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    torch.manual_seed(args.pgd_seed)
    np.random.seed(args.pgd_seed)
    random.seed(args.pgd_seed)

    model.eval()
    top1_meter = AverageMeter('PGD Acc@1')
    top5_meter = AverageMeter('PGD Acc@5')
    total = 0

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        delta = torch.empty_like(images).uniform_(-1.0, 1.0)
        delta = delta * eps
        delta = torch.clamp(delta, -eps, eps)
        delta.requires_grad_(True)

        for _ in range(args.pgd_steps):
            adv = images + delta
            adv = torch.max(torch.min(adv, upper), lower)
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                output = model(adv)
                loss = criterion(output, target)
            loss.backward()
            grad = delta.grad.detach()
            delta.data = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps)
            adv = torch.max(torch.min(images + delta, upper), lower)
            delta.data = adv - images
            delta.grad.zero_()

        adv = torch.max(torch.min(images + delta, upper), lower)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
            output = model(adv)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1_meter.update(acc1.item(), images.size(0))
        top5_meter.update(acc5.item(), images.size(0))
        total += images.size(0)

    print(f" * (ImageNet-PGD) Acc@1 {top1_meter.avg:.3f} Acc@5 {top5_meter.avg:.3f}")

    log_dir = args.pgd_log_dir if args.pgd_log_dir else args.save
    tag = args.pgd_tag.strip() if args.pgd_tag else f"{os.path.basename(os.path.normpath(args.save))}_pgd"
    out_dir = os.path.join(log_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        "pgd": {
            "top1": float(top1_meter.avg),
            "top5": float(top5_meter.avg),
            "eps": float(args.pgd_eps),
            "alpha": float(args.pgd_alpha),
            "steps": int(args.pgd_steps),
            "norm": args.pgd_norm,
            "seed": int(args.pgd_seed),
            "num_samples": int(total),
            "batch_size": args.pgd_batch_size,
        }
    }
    with open(os.path.join(out_dir, args.pgd_results_name), 'w') as f:
        json.dump(metrics, f, indent=2)

# =========================================================
# WideResNet (for ImageNet; wrn40_4)
# =========================================================
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        residual = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + residual


class WideResNet(nn.Module):
    def __init__(self, depth=40, widen_factor=4, num_classes=1000, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        widths = [16, 16 * k, 32 * k, 64 * k]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = self._make_block(widths[1], n, stride=1, drop_rate=drop_rate)
        self.block2 = self._make_block(widths[2], n, stride=2, drop_rate=drop_rate)
        self.block3 = self._make_block(widths[3], n, stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_block(self, out_planes, num_layers, stride, drop_rate):
        layers = []
        for i in range(num_layers):
            block_stride = stride if i == 0 else 1
            layers.append(WideBasicBlock(self.in_planes, out_planes, block_stride, drop_rate))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

# =========================================================
# モデル構築
# =========================================================
def build_model(arch: str, pretrained: bool, num_classes: int):
    """
    - torchvision ResNet: resnet18/34/50/101/152 など
    - torchvision ViT: vit_b_16（あれば）
    - timm ViT: vit_tiny_patch16_224 / vit_base_patch16_224 など
    """
    arch = arch.lower()

    # --- Wide ResNet (custom wrn40_4) ---
    if arch in ('wrn40_4', 'wide_resnet40_4'):
        if pretrained:
            raise ValueError("wrn40_4 does not provide pretrained weights.")
        return WideResNet(depth=40, widen_factor=4, num_classes=num_classes)

    # --- ResNet (torchvision) ---
    if arch.startswith('resnet'):
        weights = None
        # 0.13+ の weights Enum に対応
        try:
            if pretrained:
                enum_name = f"ResNet50_Weights" if arch == 'resnet50' else None
                if arch == 'resnet18':
                    weights = tvm.ResNet18_Weights.IMAGENET1K_V1
                elif arch == 'resnet34':
                    weights = tvm.ResNet34_Weights.IMAGENET1K_V1
                elif arch == 'resnet50':
                    weights = tvm.ResNet50_Weights.IMAGENET1K_V2
                elif arch == 'resnet101':
                    weights = tvm.ResNet101_Weights.IMAGENET1K_V2
                elif arch == 'resnet152':
                    weights = tvm.ResNet152_Weights.IMAGENET1K_V2
        except AttributeError:
            weights = 'IMAGENET1K_V1' if pretrained else None  # 古い torchvision 互換

        ctor = getattr(tvm, arch)  # e.g., tvm.resnet50
        m = ctor(weights=weights) if pretrained else ctor(weights=None)

        # 全結合付け替え
        # 分類ヘッドは「クラス数を変える時」か「明示的にリセット指示がある時」だけ付け替える
        parsed_args = getattr(argparse, '_parsed_args', None)
        reset_head_flag = getattr(parsed_args, 'reset_head', False) if parsed_args else False
        need_reset = (num_classes != 1000) or (not pretrained) or reset_head_flag
        if hasattr(m, 'fc') and isinstance(m.fc, nn.Linear) and need_reset:
            in_features = m.fc.in_features
            m.fc = nn.Linear(in_features, num_classes)

        return m

    # --- torchvision ViT (vit_b_16) ---
    if arch in ('vit_b_16', 'vit_b16', 'vit_b_16_tv', 'vit_base'):
        try:
            weights = None
            if pretrained and hasattr(tvm, 'ViT_B_16_Weights'):
                weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1
            m = tvm.vit_b_16(weights=weights)
            parsed_args = getattr(argparse, '_parsed_args', None)
            reset_head_flag = getattr(parsed_args, 'reset_head', False) if parsed_args else False
            need_reset = (num_classes != 1000) or (not pretrained) or reset_head_flag
            if need_reset:
                in_features = m.heads.head.in_features
                m.heads.head = nn.Linear(in_features, num_classes)

            return m
        except Exception:
            # 下の timm fallback へ
            pass

    # --- timm ViT fallback ---
    if arch in ('vit_tiny', 'vit_base', 'vit_tiny_patch16_224', 'vit_base_patch16_224'):
        try:
            import timm
        except ImportError as _e:
            raise RuntimeError(
                "timm が見つかりません。仮想環境で `pip install 'timm>=0.9'` を実行してください"
            )
        model_name = {
            'vit_tiny': 'vit_tiny_patch16_224',
            'vit_base': 'vit_base_patch16_224',
            'vit_tiny_patch16_224': 'vit_tiny_patch16_224',
            'vit_base_patch16_224': 'vit_base_patch16_224',
        }[arch]
        m = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        return m

    # --- その他 torchvision モデル（文字列指定） ---
    if hasattr(tvm, arch):
        ctor = getattr(tvm, arch)
        try:
            m = ctor(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            m = ctor(pretrained=pretrained)
        # 分類ヘッドが Linear の場合は付け替え（汎用）
        if hasattr(m, 'fc') and isinstance(m.fc, nn.Linear):
            in_features = m.fc.in_features
            m.fc = nn.Linear(in_features, num_classes)
        elif hasattr(m, 'classifier'):
            # e.g., MobileNetV3, EfficientNet 等
            if isinstance(m.classifier, nn.Sequential) and isinstance(m.classifier[-1], nn.Linear):
                in_features = m.classifier[-1].in_features
                m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m

    raise ValueError(f"Unknown arch: {arch}")

if __name__ == "__main__":
    main()
