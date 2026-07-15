from fastapi import APIRouter, HTTPException

from ..container_store import get_registry
from ..models import AssignDocumentRequest, Container, ContainerNameRequest

router = APIRouter(prefix="/containers", tags=["containers"])


@router.post("", response_model=Container)
def create_container(request: ContainerNameRequest):
    return get_registry().create(request.name)


@router.get("", response_model=list[Container])
def list_containers():
    return get_registry().list_containers()


@router.get("/{container_id}", response_model=Container)
def get_container(container_id: str):
    container = get_registry().get(container_id)
    if container is None:
        raise HTTPException(status_code=404, detail="Container not found")
    return container


@router.patch("/{container_id}", response_model=Container)
def rename_container(container_id: str, request: ContainerNameRequest):
    container = get_registry().rename(container_id, request.name)
    if container is None:
        raise HTTPException(status_code=404, detail="Container not found")
    return container


@router.delete("/{container_id}")
def delete_container(container_id: str):
    if not get_registry().delete(container_id):
        raise HTTPException(status_code=404, detail="Container not found")
    return {"deleted": True}


@router.post("/{container_id}/documents", response_model=Container)
def add_document_to_container(container_id: str, request: AssignDocumentRequest):
    """File a document into a container (many-to-many, idempotent)."""
    registry = get_registry()
    if not registry.add_document(container_id, request.doc_id):
        raise HTTPException(status_code=404, detail="Container not found")
    return registry.get(container_id)


@router.delete("/{container_id}/documents/{doc_id}")
def remove_document_from_container(container_id: str, doc_id: str):
    """Non-destructively remove a document from one container.

    The document itself is untouched -- it stays in any other containers and
    remains queryable under the virtual "All" view.
    """
    if not get_registry().remove_document(container_id, doc_id):
        raise HTTPException(
            status_code=404, detail="Document is not a member of that container"
        )
    return {"removed": True}
