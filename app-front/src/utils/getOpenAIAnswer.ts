import type { TestCSVInterface } from '../models/models';
import { client } from './client';

export const getOpenAIAnswer = async (data: TestCSVInterface) => {
  const completions = await client.chat.completions.create({
    model: 'ft:gpt-4o-mini-2024-07-18:personal:kuaf:BdCKIwgY',
    messages: [{ role: 'user', content: JSON.stringify(data) }],
  });
  console.log(JSON.stringify(data));
  const content = completions.choices[0].message.content;
  if (content === null) return '';
  const stack = content.split(' ')[1];

  return stack.slice(1, -1);
};
