import {
  isLanguage,
  languages,
  type GitHubUser,
  type LanguageCount,
  type TestCSVInterface,
  type UsersDataWithRepos,
} from '../models/models';
import { getGitHubReposList } from './getGitHubReposList';
import getRequestHeader from './getRequestHeader';

const getUser = async (
  username: string
): Promise<GitHubUser | { message: string }> => {
  const response = await fetch(
    `https://api.github.com/users/${username}`,
    getRequestHeader()
  );
  return response.json();
};

export const getUserData = async (username: string) => {
  const user = await getUser(username);
  if (typeof user === 'object' && 'message' in user) {
    return { message: user.message };
  }
  const repos = await getGitHubReposList(user.repos_url);
  const filteredRepos = repos.filter(
    repo =>
      repo.size > 500 && repo.language !== null && isLanguage(repo.language)
  );
  if (filteredRepos.length < 3) {
    return { message: '레포지토리가 너무 적어요' };
  }

  const data = {
    ...user,
    repos: filteredRepos,
  } as UsersDataWithRepos;

  let userText = '';
  const languageCount: LanguageCount = {
    Assembly: 0,
    C: 0,
    'C++': 0,
    'C#': 0,
    Dart: 0,
    Go: 0,
    Java: 0,
    JavaScript: 0,
    Kotlin: 0,
    MATLAB: 0,
    PHP: 0,
    Python: 0,
    Ruby: 0,
    Rust: 0,
    Scala: 0,
    Swift: 0,
    TypeScript: 0,
  };

  for (const repo of data.repos) {
    if (repo.language && isLanguage(repo.language)) {
      userText += repo.name.replace(/,/g, '&') + ' :: ';
      userText += repo.description
        ? repo.description.replace(/,/g, '&').replace(/\n/g, ' ') + ' / '
        : ' / ';
      languageCount[repo.language] += 1;
    }
  }
  const totalCount = Object.values(languageCount).reduce(
    (acc, curr) => acc + curr,
    0
  );

  for (const lang of languages) {
    languageCount[lang] = Number((languageCount[lang] / totalCount).toFixed(3));
  }

  const result: TestCSVInterface = {
    username: data.login,
    ...languageCount,
    text: userText,
  };

  return result;
};
