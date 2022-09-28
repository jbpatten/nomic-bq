import uuid

from nomic import AtlasClient
import pytest
import random
import time
import numpy as np

def test_map_embeddings_with_errors():
    atlas = AtlasClient()

    num_embeddings = 10
    embeddings = np.random.rand(num_embeddings, 10)


    #test nested dictionaries
    with pytest.raises(Exception):
        data = [{'hello': {'hello'}} for i in range(len(embeddings))]
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST',
                                        id_field='id',
                                        is_public=True)

    #test underscore
    with pytest.raises(Exception):
        data = [{'__hello': {'hello'} } for i in range(len(embeddings))]
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST',
                                        id_field='id',
                                        is_public=True)

    #test non-matching keys across metadatums
    with pytest.raises(Exception):
        data = [{'hello': 'a'} for i in range(len(embeddings))]
        data[1]['goodbye'] = 'b'
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST',
                                        id_field='id',
                                        is_public=True)

    #test duplicate keys error
    with pytest.raises(Exception):
        data = [{'id': 'a'} for i in range(len(embeddings))]
        data[1]['goodbye'] = 'b'
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST',
                                        id_field='id',
                                        is_public=True)

    #fail on to large metadata
    with pytest.raises(Exception):
        embeddings = np.random.rand(1000, 10)
        data = [{'id': i, 'string': ''.join(['a'] * (1048576 // 10))} for i in range(len(embeddings))]
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST',
                                        id_field='id',
                                        is_public=True)


def test_map_embeddings():
    atlas = AtlasClient()

    num_embeddings = 10
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'id': str(uuid.uuid4())} for i in range(len(embeddings))]

    response = atlas.map_embeddings(embeddings=embeddings,
                                    map_name='UNITTEST',
                                    data=data,
                                    id_field='id',
                                    is_public=True)

    assert response['project_id']
    data = [{'id': str(uuid.uuid4())} for i in range(len(embeddings))]
    while True:
        # Code executed here
        time.sleep(10)
        if atlas.is_project_accepting_data(project_id=response['project_id']):
            atlas.update_maps(project_id=response['project_id'], data=data, embeddings=embeddings)
            break

    atlas.delete_project(project_id=response['project_id'])


def test_map_text_errors():
    atlas = AtlasClient()

    # no indexed field
    with pytest.raises(Exception):
        response = atlas.map_text(data=[{'key': 'a'}],
                                  id_field='id',
                                  indexed_field='text',
                                  is_public=True,
                                  map_name='UNITTEST',
                                  map_description='test map description',
                                  num_workers=1)

    # assert empty value for a field
    with pytest.raises(Exception):
        response = atlas.map_text(data=[{'key': ''}],
                                  id_field='id',
                                  indexed_field='key',
                                  is_public=True,
                                  map_name='UNITTEST',
                                  map_description='test map description',
                                  num_workers=1)

