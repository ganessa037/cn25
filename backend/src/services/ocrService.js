export async function processDocuments(documents) {
    // documents: [{ buffer/base64/path, type }, ...]
    return {
      extractedData: {}, // 先返回空对象，避免 null 访问错误
      confidence: 0
    };
  }