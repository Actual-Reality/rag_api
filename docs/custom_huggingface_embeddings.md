# Custom HuggingFace Embeddings

This document explains how to use the custom HuggingFace embeddings provider with your own inference endpoint.

## Overview

The custom HuggingFace embeddings provider allows you to use your own deployed HuggingFace models through an inference endpoint. This is useful when you have specific models deployed on HuggingFace Inference Endpoints or other compatible services.

## Configuration

To use the custom HuggingFace embeddings provider, you need to set the following environment variables:

```env
EMBEDDINGS_PROVIDER=custom_huggingface
CUSTOM_HF_ENDPOINT=https://your-endpoint-url.huggingface.cloud
CUSTOM_HF_API_TOKEN=your-api-token
```

### Environment Variables

- `EMBEDDINGS_PROVIDER`: Set to `custom_huggingface` to use this provider
- `CUSTOM_HF_ENDPOINT`: The URL of your HuggingFace inference endpoint
- `CUSTOM_HF_API_TOKEN`: The API token for authentication with your endpoint

## Usage

Once configured, the application will automatically use your custom HuggingFace endpoint for generating embeddings. The implementation includes:

1. **Retry Logic**: Automatic retry with exponential backoff for failed requests
2. **Rate Limiting Handling**: Automatic handling of rate limiting (HTTP 429) with appropriate delays
3. **Error Handling**: Comprehensive error handling and logging
4. **Timeout Handling**: Configurable timeout for requests

## Implementation Details

The custom implementation is located in `app/services/custom_hf_embeddings.py` and provides:

- `embed_documents`: Embeds a list of documents
- `embed_query`: Embeds a single query string
- Automatic retry logic with exponential backoff
- Proper error handling for various HTTP status codes

## Example Configuration

```env
EMBEDDINGS_PROVIDER=custom_huggingface
CUSTOM_HF_ENDPOINT=https://qva6y3xao7a22kwb.us-east4.gcp.endpoints.huggingface.cloud
CUSTOM_HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Testing

To test the integration, you can use the provided test script:

```bash
python test_custom_hf_embeddings.py
```

This script will verify that your endpoint is properly configured and accessible.