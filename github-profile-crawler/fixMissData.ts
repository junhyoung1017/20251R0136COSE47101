import fs from 'fs';
import { CSVInterface, languages } from './types';

async function main() {
  console.log('🚀 틀린 라벨링을 고쳐요!');

  const file = fs.readFileSync(
    'results/github_profiles_total_v4.3.csv',
    'utf8'
  );
  const missData = fs.readFileSync('results/miss_data.csv', 'utf8');

  if (file === undefined || file.length === 0) {
    console.error('파일이 없어요!');
    return;
  }
  if (missData === undefined || missData.length === 0) {
    console.error('파일이 없어요!');
    return;
  }
  const lines = file.split('\n');
  const missDataLines = missData.split('\n');

  let csvContent = `user_ID, username, repo_count, ${languages.join(
    ', '
  )}, text, stack, note\n`;

  for (const line of lines.slice(1)) {
    const lineData = line.split(',');
    const userID = lineData[0];

    const missDataLine = missDataLines.find(
      line => line.split(',')[1] === userID
    );

    if (missDataLine) {
      const missDataLineData = missDataLine.split(',');
      lineData[21] = missDataLineData[22];
    }

    csvContent += `${lineData.join(',')}\n`;
  }

  fs.writeFileSync('results/github_profiles_total_v5.csv', csvContent);
}

main();
