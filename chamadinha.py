import cv2
import face_recognition
import datetime
import csv

# Inicialize a captura de vídeo da câmera
cap = cv2.VideoCapture(0)

# Carregue as imagens das pessoas que você deseja reconhecer
pessoa1_image = face_recognition.load_image_file("Chamada/fredstaib.png")
pessoa1_face_encoding = face_recognition.face_encodings(pessoa1_image)[0]

pessoa2_image = face_recognition.load_image_file("Chamada/caua.png")
pessoa2_face_encoding = face_recognition.face_encodings(pessoa2_image)[0]

known_face_encodings = [pessoa1_face_encoding, pessoa2_face_encoding]
known_face_names = ["Fred Staib", "Caua Almeida"]

# Dicionário para rastrear os horários de registro
registro = {}

# Nome do arquivo CSV para salvar os registros
csv_filename = "registros.csv"

while True:
    # Capture um frame da câmera
    ret, frame = cap.read()

    # Encontre todas as faces no frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Verifique se há correspondência com alguma das faces conhecidas
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Desconhecido"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Registre o horário de registro
            if name not in registro:
                registro[name] = datetime.datetime.now()

                # Salve o registro no arquivo CSV
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, registro[name]])

        # Desenhe um retângulo ao redor da face e exiba o nome da pessoa ou "OK" se já foi registrada
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, "OK" if name in registro else name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Exiba o frame com as detecções
    cv2.imshow("Reconhecimento Facial", frame)

    # Saia do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a captura de vídeo e feche todas as janelas
cap.release()
cv2.destroyAllWindows()
