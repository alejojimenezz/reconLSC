import mediapipe as mp
import math

def distancia(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def hand_map(landmarks):
    HandLandmark = mp.solutions.hands.HandLandmark
    return {
        'wrist': landmarks[HandLandmark.WRIST],
        'thumb_cmc': landmarks[HandLandmark.THUMB_CMC],
        'thumb_cmc': landmarks[HandLandmark.THUMB_MCP],
        'thumb_ip': landmarks[HandLandmark.THUMB_IP],
        'thumb_tip': landmarks[HandLandmark.THUMB_TIP],
        'index_mcp': landmarks[HandLandmark.INDEX_FINGER_MCP],
        'index_pip': landmarks[HandLandmark.INDEX_FINGER_PIP],
        'index_dip': landmarks[HandLandmark.INDEX_FINGER_DIP],
        'index_tip': landmarks[HandLandmark.INDEX_FINGER_TIP],
        'middle_mcp': landmarks[HandLandmark.MIDDLE_FINGER_MCP],
        'middle_pip': landmarks[HandLandmark.MIDDLE_FINGER_PIP],
        'middle_dip': landmarks[HandLandmark.MIDDLE_FINGER_DIP],
        'middle_tip': landmarks[HandLandmark.MIDDLE_FINGER_TIP],
        'ring_mcp': landmarks[HandLandmark.RING_FINGER_MCP],
        'ring_pip': landmarks[HandLandmark.RING_FINGER_PIP],
        'ring_dip': landmarks[HandLandmark.RING_FINGER_DIP],
        'ring_tip': landmarks[HandLandmark.RING_FINGER_TIP],
        'pinky_mcp': landmarks[HandLandmark.PINKY_MCP],
        'pinky_pip': landmarks[HandLandmark.PINKY_PIP],
        'pinky_dip': landmarks[HandLandmark.PINKY_DIP],
        'pinky_tip': landmarks[HandLandmark.PINKY_TIP]          
    }

def letra_a(landmarks):
    p = hand_map(landmarks)
    return (
        distancia(p['thumb_tip'], p['index_mcp']) < 0.1 and
        p['index_tip'].y > p['index_pip'].y and
        p['middle_tip'].y > p['middle_pip'].y and
        p['ring_tip'].y > p['ring_pip'].y and
        p['pinky_tip'].y > p['pinky_pip'].y
    )

def letra_b(landmarks):
    p = hand_map(landmarks)
    return (
        distancia(p['thumb_tip'], p['wrist']) < 0.25 and
        p['index_tip'].y < p['index_pip'].y and
        p['middle_tip'].y < p['middle_pip'].y and
        p['ring_tip'].y < p['ring_pip'].y and
        p['pinky_tip'].y < p['pinky_pip'].y
    )

def letra_c(landmarks):
    p = hand_map(landmarks)
    return (
        distancia(p['index_tip'], p['wrist']) < 0.3 and
        distancia(p['middle_tip'], p['wrist']) < 0.3
    )

def letra_d(landmarks):
    p = hand_map(landmarks)
    return (
        distancia(p['thumb_tip'], p['middle_tip']) < 0.1 and
        p['index_tip'].y < p['index_pip'].y and
        p['middle_tip'].y > p['middle_pip'].y and 
        p['ring_tip'].y > p['ring_pip'].y and 
        p['pinky_tip'].y > p['pinky_pip'].y
    )

def letra_e(landmarks):
    p = hand_map(landmarks)
    return (
        distancia(p['thumb_tip'], p['wrist']) < 0.25 and
        p['index_tip'].y > p['index_pip'].y and
        p['middle_tip'].y > p['middle_pip'].y and
        p['ring_tip'].y > p['ring_pip'].y and
        p['pinky_tip'].y > p['pinky_pip'].y
    )

def letra_f(landmarks):
    p = hand_map(landmarks)
    return (
        distancia(p['thumb_tip'], p['index_mcp']) < 0.1 and
        p['index_tip'].y < p['index_pip'].y and
        p['middle_tip'].y > p['middle_pip'].y and
        p['ring_tip'].y > p['ring_pip'].y and
        p['pinky_tip'].y > p['pinky_pip'].y
    )

def letra_i(landmarks):
    p = hand_map(landmarks)
    return (
        p['thumb_tip'].x > p['thumb_ip'].x and
        p['index_tip'].y > p['index_pip'].y and
        p['middle_tip'].y > p['middle_pip'].y and
        p['ring_tip'].y > p['ring_pip'].y and
        p['pinky_tip'].y < p['pinky_pip'].y
    )

def letra_l(landmarks):
    p = hand_map(landmarks)
    return (
        distancia(p['thumb_tip'], p['middle_tip']) > 0.15 and
        p['thumb_tip'].x < p['thumb_ip'].x and
        p['index_tip'].y < p['index_pip'].y and
        p['middle_tip'].y > p['middle_pip'].y and
        p['ring_tip'].y > p['ring_pip'].y and
        p['pinky_tip'].y > p['pinky_pip'].y
    )

static_alphabet = {
    "A": letra_a,
    "B": letra_b,
    "C": letra_c,
    "D": letra_d,
    "E": letra_e,
    "F": letra_f,
    "I": letra_i,
    # "K": letra_k,
    "L": letra_l,
    # "M": letra_m,
    # "N": letra_n,
    # "O": letra_o,
    # "P": letra_p,
    # "Q": letra_q,
    # "R": letra_r,
    # "T": letra_t,
    # "U": letra_u,
    # "V": letra_v,
    # "W": letra_w,
    # "X": letra_x,
    # "Y": letra_y,
}

dynamic_alphabet = {
    # "G": letra_g,
    # "H": letra_h,
    # "J": letra_j,
    # "NN": letra_nn,
    # "S": letra_s,
    # "Z": letra_z,
}