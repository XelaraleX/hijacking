from models.yolov3.yolov3_wrapper import YOLOv3
from models.yolov3.yolov3_model import yolo_head, yolo_correct_boxes
from keras import backend as K
import tensorflow as tf
from utils.image_utils import load_yolov3_image
import numpy as np
import cv2

save_img_with_bboxes = False

image_shape = [416, 416]
target_class = 2
bbox_loc = [106, 248, 231, 325] # left, top, right, bottom
patch_coords = [188, 284, 247, 309]
patch_mask = np.zeros(np.array([416, 416]), np.float32)
patch_mask[patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2]] = 1.
patch_mask = np.concatenate([[patch_mask]] * 3, axis=0)
patch_mask = np.moveaxis(patch_mask, 0, -1)
patch_mask = np.expand_dims(patch_mask, axis=0)
print(patch_mask)
patch_unmask = 1. - patch_mask
print(patch_mask.sum())
lambda_ = 1
cx, cy = (bbox_loc[2] + bbox_loc[0]) // 2, (bbox_loc[3] + bbox_loc[1]) // 2

sess = K.get_session()
model = YOLOv3(sess=sess)
num_layers = len(model.model.output)
image = load_yolov3_image('output/ori_3.png')
patch = patch_mask * image
print(patch.sum())
cv2.imwrite(f'./output/patch/patch_{0}.png', patch[0])
input_shape = K.shape(model.model.output[0])[1:3] * 32
anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

box_class_probs_logits_t = {}
bbox_contains_mask = {}

if save_img_with_bboxes:
    image_ = image[0] * 255
    image_ = image_[:, :, [2, 1, 0]]
    image_bbox = image_.copy()

for l in range(num_layers):
    box_xy, box_wh, box_confidence, box_class_probs, box_coord_logits,\
    box_confidence_logits, box_class_probs_logits = yolo_head(model.model.output[l], 
                    model.anchors[anchor_mask[l]], model.num_classes, 
                    input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)

    box_xy, box_wh, box_confidence, box_class_probs, box_coord_logits,\
    box_confidence_logits, box_class_probs_logits, boxes = sess.run([box_xy, box_wh,
        box_confidence, box_class_probs, box_coord_logits,
        box_confidence_logits, box_class_probs_logits, boxes], feed_dict={
            model.input_image_: image,
            model.input_image_shape: image_shape,
            model.patch: patch,
            model.patch_mask: patch_mask,
            K.learning_phase(): 0,
        })

    print(box_xy.shape, box_wh.shape, box_confidence.shape,
          box_class_probs.shape, box_coord_logits.shape,
          box_confidence_logits.shape, box_class_probs_logits.shape)

    boxes = np.reshape(np.array(boxes), [-1, 4])
    bbox_contains_mask[l] = np.zeros(boxes.shape[0], np.int32)

    for i, bbox in enumerate(boxes):
        if cx > bbox[0] and cx < bbox[2] and cy > bbox[1] and cy < bbox[3]:
            bbox_contains_mask[l][i] = 1

            if save_img_with_bboxes:
                left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                image_bbox = cv2.rectangle(image_bbox, (left, top), (right, bottom), (0, 0, 255), 1)

    box_class_probs_logits = tf.reshape(box_class_probs_logits, [-1, 80])
    box_class_probs_logits_t[l] = np.zeros(box_class_probs_logits.shape, np.float32)
    box_class_probs_logits_t[l][np.where(bbox_contains_mask[l] == 1), target_class] = 1

if save_img_with_bboxes:
    image_bbox = cv2.circle(image_bbox, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imwrite('./output/' + 'filtered_bboxes' + '.png', image_bbox)

bbox_contains_mask_ = {}
box_class_probs_logits_ = {}
for l, size in enumerate([507, 2028, 8112]):
    print(l, size)
    bbox_contains_mask_[l] =  tf.placeholder(tf.float32, shape=[size], name=f'contains_mask_{l}')
    box_class_probs_logits_[l] = tf.placeholder(tf.float32, shape=[size, 80], name=f'box_class_probs_logits_{l}')
bbox_loc_ = tf.placeholder(tf.float32, shape=[4], name='bbox_loc')
#patch_mask_ =  tf.placeholder(bool, shape=patch_mask.shape, name=f'patch_mask')

cx_, cy_ = (bbox_loc_[2] + bbox_loc_[0]) // 2, (bbox_loc_[3] + bbox_loc_[1]) // 2
w_ = tf.sqrt(bbox_loc_[2] - bbox_loc_[0])
h_ = tf.sqrt(bbox_loc_[3] - bbox_loc_[1])

loss = 0
optimizer = tf.compat.v1.train.AdamOptimizer()

for l in range(num_layers):
    print(model.input_image)
    print(patch_mask.shape)

    mbox_xy, mbox_wh, mbox_confidence, mbox_class_probs, mbox_coord_logits,\
    mbox_confidence_logits, mbox_class_probs_logits = yolo_head(
        feats=model.model.output[l], 
        anchors=model.anchors[anchor_mask[l]],
        num_classes=model.num_classes, 
        input_shape=input_shape
    )

    mbox_xy = tf.reshape(mbox_xy, [-1, 2])
    mbox_wh = tf.reshape(mbox_wh, [-1, 2])
    mbox_confidence_logits = tf.reshape(mbox_confidence_logits, [-1, 1])
    mbox_class_probs_logits = tf.reshape(mbox_class_probs_logits, [-1, 80])

    # mbox_class_probs_logits = tf.nn.softmax(mbox_class_probs_logits)
    # cross_entropy = -tf.reduce_sum(box_class_probs_logits_[l] * tf.math.log(mbox_class_probs_logits), axis=[1])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=mbox_class_probs_logits, labels=box_class_probs_logits_[l])
    loss_1 = tf.reduce_sum((mbox_confidence_logits ** 2 - cross_entropy) * bbox_contains_mask_[l], name=f'loss_1_{l}')

    boxes = yolo_correct_boxes(mbox_xy, mbox_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    cx = (boxes[:, 0] + boxes[:, 2]) // 2
    cy = (boxes[:, 1] + boxes[:, 3]) // 2
    w = tf.sqrt(boxes[:, 2] - boxes[:, 0])
    h = tf.sqrt(boxes[:, 3] - boxes[:, 1])

    centre_loss = (tf.math.subtract(cx, cx_) ** 2 + tf.math.subtract(cy, cy_) ** 2)
    size_loss = tf.math.subtract(w, w_) ** 2 + tf.math.subtract(h, h_) ** 2
    loss_2 = tf.reduce_sum(((1 - mbox_confidence_logits) ** 2 + centre_loss + size_loss +\
        cross_entropy) * bbox_contains_mask_[l], name=f'loss_2_{l}')

    loss += loss_1 + lambda_ * loss_2

grad = tf.gradients(loss, [model.patch])
print(grad)
print(model.patch)
patch_ = model.patch - 0.001 * tf.sign(grad)[0] * patch_mask
    #patch = optimizer.minimize(loss, var_list=[model.patch])
    #grad = tf.gradients(loss, model.input_image)
    #patch_ = optimizer.minimize(loss, var_list=[model.patch])
print(patch_)
adv_x = patch_unmask * model.input_image + patch_

print(patch.sum())
patience = 5
for it in range(50001):
    cur_loss, patch, adv, gradient = sess.run([loss, patch_, adv_x, grad], feed_dict={
            model.input_image_: image,
            model.input_image_shape: [416, 416],
            bbox_loc_: bbox_loc,
            model.patch: patch,
            model.patch_mask: patch_unmask,
            box_class_probs_logits_[0]: box_class_probs_logits_t[0],
            box_class_probs_logits_[1]: box_class_probs_logits_t[1],
            box_class_probs_logits_[2]: box_class_probs_logits_t[2],
            bbox_contains_mask_[0]: bbox_contains_mask[0],
            bbox_contains_mask_[1]: bbox_contains_mask[1],
            bbox_contains_mask_[2]: bbox_contains_mask[2],
            K.learning_phase(): 0,
        })
    print(patch.sum())
    print(f'{it}: {cur_loss}')
    image = adv#[0]#[0]
    adv = (image * 255)[0][:, :, [2, 1, 0]]

    if it and cur_loss >= best_loss:
        patience -= 1
        if not patience:
            cv2.imwrite(f'./output/adv_{it}.png', adv)
            break
    else:
        patience = 5
    #cv2.imwrite(f'./output/patch/patch_{it}.png', patch[0])
    best_loss = cur_loss

    if it % 1000 == 0:
        cv2.imwrite(f'./output/adv_{it}.png', adv)

cv2.imwrite(f'./output/adv_{it}.png', adv)
