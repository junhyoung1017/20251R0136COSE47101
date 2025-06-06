// TODO: 긁어올 데이터 정하기
export const languages = [
  'Assembly',
  'C',
  'C++',
  'C#',
  'Dart',
  'Go',
  'Java',
  'JavaScript',
  'Kotlin',
  'MATLAB',
  'PHP',
  'Python',
  'Ruby',
  'Rust',
  'Scala',
  'Swift',
  'TypeScript',
] as const;

export type Language = (typeof languages)[number];

export type LanguageCount = {
  [key in Language]: number;
};

export interface CSVInterface extends LanguageCount {
  username: string;
  userID: number;
  repoCount: number;
  text: string;
  stack: string;
  note: string;
}

export interface AwesomeUserInterface {
  username: string;
  userID: number;
  repoCount: number;
  stack: string;
}

export interface TestCSVInterface extends LanguageCount {
  username: string;
  userID: number;
  repoCount: number;
  text: string;
}

export interface GitHubUser {
  name?: string | null;
  email?: string | null;
  login: string;
  id: number;
  node_id: string;
  avatar_url: string;
  gravatar_id: string | null;
  url: string;
  html_url: string;
  followers_url: string;
  following_url: string;
  gists_url: string;
  starred_url: string;
  subscriptions_url: string;
  organizations_url: string;
  repos_url: string;
  events_url: string;
  received_events_url: string;
  type: string;
  site_admin: boolean;
  starred_at?: string;
  user_view_type?: string;
}

export type GitHubUserResponse = GitHubUser[];

export interface GitHubRepository {
  id: number;
  node_id: string;
  name: string;
  full_name: string;
  private: boolean;
  owner: GitHubUser;
  html_url: string;
  description: string | null;
  fork: boolean;
  url: string;
  created_at: string;
  updated_at: string;
  pushed_at: string;
  git_url: string;
  ssh_url: string;
  clone_url: string;
  size: number;
  stargazers_count: number;
  watchers_count: number;
  language: string | null;
  has_issues: boolean;
  has_projects: boolean;
  has_downloads: boolean;
  has_wiki: boolean;
  has_pages: boolean;
  has_discussions: boolean;
  forks_count: number;
  archived: boolean;
  disabled: boolean;
  open_issues_count: number;
  license: {
    key: string;
    name: string;
    url: string;
    spdx_id: string;
    node_id: string;
  } | null;
  allow_forking: boolean;
  is_template: boolean;
  web_commit_signoff_required: boolean;
  topics: string[];
  visibility: string;
  forks: number;
  watchers: number;
  default_branch: string;
}

export type UsersDataWithRepos = GitHubUser & {
  repos: GitHubRepository[];
};

export const isLanguage = (language: string): language is Language => {
  return languages.includes(language as Language);
};
