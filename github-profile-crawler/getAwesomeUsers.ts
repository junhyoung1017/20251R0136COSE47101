import fs from 'fs';
import { AwesomeUserInterface } from './types';

async function getAwesomeUsers() {
  console.log('🚀 추천 유저 풀을 만들어요!');

  const file = fs.readFileSync('results/github_profiles_total_v5.csv', 'utf8');

  if (file === undefined || file.length === 0) {
    console.error('파일이 없어요!');
    return;
  }
  const lines = file.split('\n');

  const result: AwesomeUserInterface[] = [];

  for (const line of lines.slice(1)) {
    const lineData = line.split(',');
    result.push({
      username: lineData[1],
      userID: Number(lineData[0]),
      repoCount: Number(lineData[2]),
      stack: lineData[21],
    });
  }

  fs.writeFileSync(
    'results/awesome_users.json',
    JSON.stringify(result, null, 2)
  );
}

getAwesomeUsers();
