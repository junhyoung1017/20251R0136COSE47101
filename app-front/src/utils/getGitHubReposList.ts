import type { GitHubRepository } from '../models/models';
import getRequestHeader from './getRequestHeader';

const fetchData = async (
  uri: string
): Promise<GitHubRepository[] | { message: string }> => {
  const response = await fetch(uri, getRequestHeader());
  if (response.status !== 200) {
    return { message: 'Failed to fetch repositories' };
  }
  return response.json();
};

export const getGitHubReposList = async (uri: string) => {
  const data = await fetchData(uri);

  if (typeof data === 'object' && 'message' in data) {
    console.log('API 호출 오류 발생: ' + data.message);
    return [];
  }

  return data;
};
