from models.yolov3.yolov3_wrapper import YOLOv3
from models.yolov3.yolov3_model import box_iou, preprocess_true_boxes, yolo_head, yolo_correct_boxes
from keras import backend as K
import tensorflow as tf
from utils.image_utils import load_yolov3_image
import numpy as np
import cv2

save_img_with_bboxes = False

image_shape = [416, 416]
target_class = 2
bbox_loc = [106, 248, 231, 325] # left, top, right, bottom
orig_bbox_loc = [139.35362243652344, 256.6859130859375, 264.1953125, 332.96710205078125]
patch_coords = [188, 284, 247, 309]
patch_mask = np.zeros(np.array(image_shape), np.float32)
patch_mask[patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2]] = 1.
patch_mask = np.concatenate([[patch_mask]] * 3, axis=0)
patch_mask = np.moveaxis(patch_mask, 0, -1)
patch_mask = np.expand_dims(patch_mask, axis=0)
patch_unmask = 1. - patch_mask
lambda_ = 1
cx, cy = (bbox_loc[2] + bbox_loc[0]) / 2, (bbox_loc[3] + bbox_loc[1]) / 2

sess = K.get_session()
model = YOLOv3(sess=sess)
num_layers = len(model.model.output)
image = load_yolov3_image('output/original_3.png')
patch = patch_mask * image
input_shape = K.shape(model.model.output[0])[1:3] * 32
anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

box_class_probs_logits_t = {}
bbox_contains_mask = {}

if save_img_with_bboxes:
    image_ = image[0] * 255
    image_ = image_[:, :, [2, 1, 0]]
    image_bbox = image_.copy()

true_boxes = [[106, 248, 231, 325, 2]]

y_true = preprocess_true_boxes(np.array([true_boxes]), image_shape, model.anchors, model.num_classes)
for i in range(3):
    bbox_loc_mask = y_true[i][..., 4].astype(bool)
    if len(y_true[i][..., :4][y_true[i][..., 4] == 1]) > 0:
        pos_mask = bbox_loc_mask
        layer = i

grid_shapes = [K.cast(K.shape(model.model.output[l])[1:3],
                          np.float32) for l in range(num_layers)]
m = K.shape(model.model.output[0])[0]  # batch size, tensor
mf = K.cast(m, K.dtype(model.model.output[0]))

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

    for j in range(boxes.shape[1]):
        for i in range(boxes.shape[2]):
            for k in range(boxes.shape[3]):
                if cx > boxes[0, j, i, k, 0] and cx < boxes[0, j, i, k, 2] and \
                    cy > boxes[0, j, i, k, 1] and cy < boxes[0, j, i, k, 3]:
                    true_boxes.append([boxes[0, j, i, k, 0], boxes[0, j, i, k, 1],
                                       boxes[0, j, i, k, 2], boxes[0, j, i, k, 3], 2])

                    if save_img_with_bboxes:
                        left, top, right, bottom = int(boxes[0, j, i, k, 0]), int(boxes[0, j, i, k, 1]), int(boxes[0, j, i, k, 2]), int(boxes[0, j, i, k, 3])
                        image_bbox = cv2.rectangle(image_bbox, (left, top), (right, bottom), (0, 0, 255), 1)

y_true = preprocess_true_boxes(np.array([true_boxes]), image_shape, model.anchors, model.num_classes)

if save_img_with_bboxes:
    image_bbox = cv2.circle(image_bbox, (int(cx), int(cy)), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imwrite('./output/' + 'filtered_bboxes2' + '.png', image_bbox)

loss = 0
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

input_shape = K.cast(K.shape(model.model.output[0])[1:3] * 32, tf.float32)

for l in range(1):
    object_mask = y_true[l][..., 4:5]
    true_class_probs = y_true[l][..., 5:]

    grid, raw_pred, pred_xy, pred_wh = yolo_head(
        model.model.output[l],
        model.anchors[anchor_mask[l]],
        model.num_classes, input_shape, calc_loss=True)

    raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
    raw_true_x, raw_true_y = raw_true_xy[pos_mask][0], raw_true_xy[pos_mask][1]

    raw_true_wh = K.log(y_true[l][..., 2:4] / model.anchors[anchor_mask[l]] * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh, dtype=tf.float32))
    box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

    class_loss = object_mask * K.binary_crossentropy(
            true_class_probs,
            raw_pred[..., 5:],
            from_logits=True)
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
    class_loss = tf.reduce_sum(class_loss, -1)
    loss_1 = tf.reduce_sum((confidence_loss[..., 0] - class_loss) * object_mask[0][..., 0], name=f'loss_1_{l}')

    centre_loss = tf.math.subtract(raw_pred[..., 0:2], raw_true_xy) ** 2
    centre_loss = centre_loss[..., 0]
    size_loss = tf.square(raw_true_wh - raw_pred[..., 2:4])
    size_loss = size_loss[..., 0]

    loss_2 = tf.reduce_sum((confidence_loss[..., 0] + centre_loss + size_loss + class_loss) * object_mask[0][..., 0], name=f'loss_2_{l}')

    loss += loss_2   

grad = tf.gradients(loss, [model.patch])
grad = optimizer.compute_gradients(loss, var_list=[model.patch])
patch_ = model.patch - 0.01 * tf.sign(grad[0][0]) * patch_mask
adv_x = patch_unmask * model.input_image + patch_

# print(patch.sum())
patience = 5
for it in range(50001):
    cur_loss, patch, adv, object_mask = sess.run([loss, patch_, adv_x, loss_2], feed_dict={
            model.input_image_: image,
            model.input_image_shape: [416, 416],
            model.patch: patch,
            model.patch_mask: patch_unmask,
            K.learning_phase(): 0,
        })
    print(f'{it}: {cur_loss}')
    print(object_mask)
    image = adv
    adv = (adv * 255)[0][:, :, [2, 1, 0]]

    if it and cur_loss >= best_loss:
        patience -= 1
        if not patience:
            cv2.imwrite(f'./output/adv_{it}.png', adv)
            break
    else:
        patience = 5

    best_loss = cur_loss

    if it % 50 == 0:
        cv2.imwrite(f'./output/adv_{it}.png', adv)

cv2.imwrite(f'./output/adv_{it}.png', adv)
