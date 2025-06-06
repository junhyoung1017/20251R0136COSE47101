const getRequestHeader = () => {
  return {
    method: 'GET',
    headers: {
      'X-GitHub-Api-Version': '2022-11-28',
      Authorization: import.meta.env.VITE_GITHUB_TOKEN
        ? `Bearer ${import.meta.env.VITE_GITHUB_TOKEN}`
        : '',
      Accept: 'application/vnd.github+json',
    },
  };
};

export default getRequestHeader;
