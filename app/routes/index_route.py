from fastapi import APIRouter, Request, Response

# Set up the router
router = APIRouter()


# Create a route
@router.get("/")
async def route(req: Request, res: Response):  # Create a function for the route
    # Create a response object (this is what will be returned as JSON)
    response = {
        "message": "Server is Up and Running",
    }

    res.status_code = 200  # Set the status code of the response
    return response  # Return the response
