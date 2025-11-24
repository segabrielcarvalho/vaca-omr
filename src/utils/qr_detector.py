import cv2


def detect_qr(image):
    """
    Detecta e decodifica um QR Code na imagem fornecida.
    Retorna um dicionário com dados e pontos ou None se não houver QR.
    """
    if image is None:
        return None

    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(image)

    if data:
        return {
            "data": data,
            "points": points.tolist() if points is not None else None,
        }
    return None
