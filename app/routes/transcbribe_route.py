from fastapi import APIRouter, File, UploadFile, Response
from tempfile import NamedTemporaryFile
import os
from ..utils.transcribe import transcribe

router = APIRouter()


@router.post("")
@router.post("/")
async def route(file: UploadFile = File(...), res: Response = Response()):
    try:
        # Take the file and write it to a temporary file
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
            transcription = await transcribe(temp_file_path)  # Transcribe the file

        # Close the temporary file and delete it
        temp_file.close()
        os.remove(temp_file_path)

        response = {"transcription": transcription}
        return response

    except Exception as error:
        print(f"Error during transcription: {str(error)}")
        response = {"error": f"Internal Server Error"}
        res.status_code = 500
        return response
