import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

// const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

// Chat History:
// {chat_history}
// Follow Up Input: {question}
// Standalone question:`;

// const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
// If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
// If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

// {context}

// Question: {question}
// Helpful answer in markdown:`;

const CONDENSE_PROMPT = `Dưới đây là một cuộc hội thoại và một câu hỏi kế tiếp trong cuộc hội thoại đó, hãy diễn đạt lại câu hỏi kế tiếp thành một câu hỏi thống nhất.

Lịch sử cuộc hội thoại:
{chat_history}
Câu hỏi kế tiếp trong cuộc hội thoại: {question}
Câu hỏi thống nhất:`;

const QA_PROMPT = `Bạn là một trợ lý AI hữu ích. Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi ở cuối.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết. KHÔNG cố gắng tạo ra một câu trả lời.
Nếu câu hỏi không liên quan đến ngữ cảnh, hãy trả lời một cách lịch sự rằng bạn được điều chỉnh để chỉ trả lời những câu hỏi liên quan đến ngữ cảnh.

{context}

Câu hỏi: {question}
Câu trả lời hữu ích dưới dạng markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
