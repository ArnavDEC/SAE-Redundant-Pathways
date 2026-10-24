from docker.models.containers import Container
from pathlib import Path
import io
import tarfile

def putfile_container(container: Container, filebytes: bytes,
                      filepath: Path) -> None:
    tar_stream = io.BytesIO()
    tar_file = tarfile.TarFile(fileobj=tar_stream, mode='w')
    file_info = tarfile.TarInfo(name=filepath.name)
    file_info.size = len(filebytes)
    tar_file.addfile(file_info, io.BytesIO(filebytes))
    tar_file.close()
    tar_stream.seek(0)
    container.put_archive(str(filepath.parent), tar_stream)

def get_container_file_contents(container: Container, filepath: Path) -> bytes:
    bits, stat = container.get_archive(str(filepath))
    with tarfile.open(fileobj=io.BytesIO(b''.join(bits))) as tf:
        member = tf.getmembers()[0]
        f = tf.extractfile(member)
        assert f is not None
        return f.read()

def copydir_to_container(container: Container, src: Path, dest: Path) -> None:
    tar_stream = io.BytesIO()
    tar = tarfile.TarFile(fileobj=tar_stream, mode='w')
    tar.add(src, arcname=src.name)
    tar.close()
    tar_stream.seek(0)
    container.put_archive(str(dest), tar_stream)
