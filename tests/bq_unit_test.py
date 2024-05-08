import unittest
from unittest.mock import patch
import pandas as pd

from nomic.atlas import map_data

class TestMapData(unittest.TestCase):

  @patch('google.cloud.bigquery.Client')
  def test_bigquery_embeddings(self, mock_client):
      # Define sample query and expected DataFrame
      sample_query = "SELECT id, features FROM my_dataset.my_table"
      expected_data = pd.DataFrame({'id': [1, 2], 'features': [[1.0, 2.0], [3.0, 4.0]]})

      # Mock BigQuery client and query methods
      mock_query = mock_client.return_value.query
      mock_query.return_value.result.to_dataframe.return_value = expected_data

      # Call map_data with mocked QueryJob and desired column
      query_job = mock_client.return_value.query(sample_query)
      embeddings_column = 'features'
      dataset = map_data(data=query_job, embeddings_column=embeddings_column)

      # Assert extracted embeddings and data
      self.assertEqual(dataset.embeddings.tolist(), expected_data[embeddings_column].tolist())
      self.assertNotIn(embeddings_column, dataset.data.columns)

if __name__ == '__main__':
  unittest.main()