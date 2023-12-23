import fastapi.responses
import numpy as np
from PIL import Image, ImageOps
import os
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
from pymongo import MongoClient
from insightface.app import FaceAnalysis
import insightface
from tqdm import tqdm

FOLDER = '/home/siva/Documents/Test/unknown/'
KNOWN_FOLDER = '/home/siva/Documents/Test/known/'
THUMBNAIL_FOLDER = '/home/siva/Documents/Test/unknown/.thumbnail/'
TMP_FOLDER = '/home/siva/Documents/Test/.tmp/'
FACE_TO_FILE_MAP = {}
client = MongoClient("localhost", 27017)
database = 'face_recognition'
unknown_data_collection = 'unknown_data'
known_faces_collection = 'known_faces'

face_insight = FaceAnalysis(providers=['CPUExecutionProvider'])
face_insight.prepare(ctx_id=0, det_size=(640, 640))
handler = insightface.model_zoo.get_model('buffalo_l')
handler.prepare(ctx_id=0)

db = client[database]


def find_faces_and_store(filename):
    img = cv2.imread(f'{TMP_FOLDER}{filename}')
    faces = face_insight.get(img)
    idx = 1

    faces_data = []
    for face in faces:
        bbox = face['bbox']
        embedding = face['embedding']

        head_height = int((bbox[3] - bbox[1]) * 1.3)  # Adjust the factor based on your preference
        margin = (head_height - (bbox[3] - bbox[1]))

        new_x = max(0, int(bbox[0] - margin))
        new_y = max(0, int(bbox[1] - margin))
        new_w = int(bbox[2] - bbox[0] + 2 * margin)
        new_h = int(bbox[3] - bbox[1] + 2 * margin)

        # Ensure the new bounding box stays within the image bounds
        new_bbox = [new_x, new_y, new_x + new_w, new_y + new_h]
        new_bbox = [max(0, val) for val in new_bbox]

        cropped_face = img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]
        face_id = f'face-{idx}-{filename}'
        cv2.imwrite(f"{KNOWN_FOLDER}{face_id}", cropped_face)
        idx = idx + 1

        faces_data.append({
            'face_id': face_id.split('.')[0],
            'embedding': embedding.tolist(),
            'image_path': f"{KNOWN_FOLDER}{face_id}",
        })
    db[known_faces_collection].insert_many(faces_data)


def create_thumbnail(file_path, file_name):
    image = Image.open(file_path)
    image = ImageOps.exif_transpose(image)
    image.thumbnail((image.width*200//image.height, 200))

    # creating thumbnail
    image.save(THUMBNAIL_FOLDER + file_name)


def load_db_mongo():
    print("loading mongodb")
    for root, dirs, files in os.walk(FOLDER):
        if 'thumbnail' in root:
            continue
        for i in tqdm(range(len(files))):
            filename = files[i]
            file_path = os.path.join(root, filename)
            if not ('.jpg' in filename.lower() or '.jpeg' in filename.lower() or '.png' in filename.lower()):
                continue
            img = cv2.imread(file_path)
            faces = face_insight.get(img)

            embeddings = []
            for face in faces:
                embeddings.append(face['embedding'].tolist())

            create_thumbnail(file_path, filename)
            db[unknown_data_collection].insert_one({
                'image_name': filename,
                'image_path': file_path,
                'embeddings': embeddings
            })


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/faces/known')
def get_known_face_list():
    known_images = db[known_faces_collection].find()
    return {
        'images':  [known['face_id'] for known in known_images]
    }

@app.get(
    '/photos/thumbnail/{image_name}',
)
def get_thumbnail(image_name: str):
    return fastapi.responses.FileResponse(THUMBNAIL_FOLDER + image_name)


@app.get(
    '/image',
)
def get_image(name: str):
    return fastapi.responses.FileResponse(FOLDER + name)

@app.get(
    '/face/{face_id}',
)
def get_face(face_id: str):
    known_face = db[known_faces_collection].find_one({
        'face_id': face_id
    })
    return fastapi.responses.FileResponse(known_face['image_path'])


@app.post('/upload')
def upload_known_face(photoFile: UploadFile):

    file_location = f"{TMP_FOLDER}{photoFile.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(photoFile.file.read())
    find_faces_and_store(photoFile.filename)
    os.remove(file_location)
    return {
        'message': 'Success'
    }


@app.get('/photos')
def get_photos(face_id: str = None):
    unknown_images = db[unknown_data_collection].find()
    result_files = []

    if not face_id:
        result_files = [unknown['image_name'] for unknown in unknown_images]
    else:
        obj = db[known_faces_collection].find_one(
            {'face_id': face_id},
        )
        known_embeddings = np.array(obj['embedding'])
        for unknown in unknown_images:
            unknown_embeddings = unknown['embeddings']
            for emb in unknown_embeddings:
                sim = handler.compute_sim(known_embeddings, np.array(emb))
                if sim >= 0.4:
                    result_files.append(unknown['image_name'])
    return {
        'images': result_files
    }


if __name__ == '__main__':
    # check_match()
    # load_db()
    # dfs = DeepFace.find('/home/siva/Documents/Test/vachu.JPG'
    #                     , db_path='/home/siva/Documents/Test/unknown/'
    #                     , model_name='Facenet'
    #                     , distance_metric='euclidean'
    #                     , detector_backend='retinaface')
    #
    # print(dfs[0].head(10))
    # load_db_mongo()
    # app = FaceAnalysis(providers=['CPUExecutionProvider'])
    # app.prepare(ctx_id=0, det_size=(640, 640))
    # handler = insightface.model_zoo.get_model('buffalo_l')
    # handler.prepare(ctx_id=0)
    # # handler.compute_sim('')
    # # handler.get()
    #
    #
    # img = cv2.imread('/home/siva/Documents/Test/vachu.JPG')
    # faces = app.get(img)
    #
    # counter = 0
    # for filename in os.scandir(FOLDER):
    #     if (filename.path == '/home/siva/Documents/Test/unknown/vector.index'
    #             or filename.path == '/home/siva/Documents/Test/unknown/file.json'):
    #         continue
    #     if filename.is_file():
    #         img2 = cv2.imread(filename.path)
    #         facesU = app.get(img2)
    #
    #         simarr = []
    #         for face in facesU:
    #             sim = handler.compute_sim(face['embedding'], faces[0]['embedding'])
    #             simarr.append(sim)
    #             # if sim > 0.6:
    #             #     print(sim)
    #             #     rimg = app.draw_on(img2, [face])
    #             #     cv2.imwrite(f"/home/siva/Documents/Test/test{counter}.jpg", rimg)
    #             #     counter = counter + 1
    #         print(f"File: {filename.name}:: {max(simarr)}")

    # print(faces)
    # rimg = app.draw_on(img, faces)
    # cv2.imwrite("/home/siva/Documents/Test/test.jpg", rimg)

    for file in os.scandir(FOLDER):
        if 'thumbnail' in file.name:
            continue
        create_thumbnail(file.path, file.name)





