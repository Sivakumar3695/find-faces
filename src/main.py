import fastapi.responses
import numpy as np
import uvicorn
from PIL import Image, ImageOps
import os
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
from pymongo import MongoClient
from insightface.app import FaceAnalysis
import insightface
from tqdm import tqdm
from utils import config_loader


config = config_loader.load()

# ---- folder constants ----
image_folder = config['image_folder']
known_folder = f"{config['image_folder']}/.known"
tmp_folder = f"{config['image_folder']}/.tmp"
thumbnail_folder = f"{config['image_folder']}/.thumbnail"
# ---- folder constantns end ----


# ---- start of DB initialization ----
database = 'face_recognition'
unknown_data_collection = 'unknown_data'
known_faces_collection = 'known_faces'

client = MongoClient(config['db_url'])
db = client[database]
# ---- end of DB initialization ----

# ---- start model initialization ----
face_insight = FaceAnalysis(providers=['CPUExecutionProvider'])
face_insight.prepare(ctx_id=0, det_size=(640, 640))
handler = insightface.model_zoo.get_model('buffalo_l')
handler.prepare(ctx_id=0)
# ---- end model initialization ----


def find_faces_and_store(filename):
    img = cv2.imread(f"{tmp_folder}/{filename}")
    faces = face_insight.get(img)
    idx = 1

    faces_data = []
    for face in faces:
        bbox = face['bbox']
        embedding = face['embedding']

        head_height = int((bbox[3] - bbox[1]) * 1.3)
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
        cv2.imwrite(f"{known_folder}/{face_id}", cropped_face)
        idx = idx + 1

        faces_data.append({
            'face_id': face_id.split('.')[0],
            'embedding': embedding.tolist(),
            'image_path': f"{known_folder}/{face_id}",
        })
    db[known_faces_collection].insert_many(faces_data)


def create_thumbnail(file_path, file_name):
    image = Image.open(file_path)
    image = ImageOps.exif_transpose(image)
    image.thumbnail((image.width*200//image.height, 200))

    # creating thumbnail
    image.save(f"{thumbnail_folder}/{file_name}", quality=95)


def load_db_mongo():
    os.makedirs(f"{config['image_folder']}/.thumbnail/", exist_ok=True)
    os.makedirs(f"{config['image_folder']}/.known/", exist_ok=True)
    os.makedirs(f"{config['image_folder']}/.tmp/", exist_ok=True)

    print("loading mongodb")

    if not os.path.exists(config['image_folder']):
        raise Exception('Path does not exist')

    for root, dirs, files in os.walk(config['image_folder']):
        if 'thumbnail' in root:
            continue
        for i in tqdm(range(len(files))):
            filename = files[i]
            file_path = os.path.join(root, filename)
            if not (
                    '.jpg' in filename.lower()
                    or '.jpeg' in filename.lower()
                    or '.png' in filename.lower()
            ):
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
    return fastapi.responses.FileResponse(f"{thumbnail_folder}/{image_name}")


@app.get(
    '/image',
)
def get_image(name: str):
    return fastapi.responses.FileResponse(f"{config['image_folder']}/{name}")


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
    file_location = f"{tmp_folder}/{photoFile.filename}"
    print(file_location)
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
    uvicorn.run("main:app", port=8000, log_level="info")
    # load_db_mongo()
    # for file in os.scandir(FOLDER):
    #     if 'thumbnail' in file.name:
    #         continue
    #     create_thumbnail(file.path, file.name)
