import fs from 'fs';
import {
  CSVInterface,
  isLanguage,
  LanguageCount,
  languages,
  UsersDataWithRepos,
} from './types';

async function main() {
  console.log('🚀 추가 정보를 가져와요!');

  const userList: UsersDataWithRepos[] = [];
  fs.readdirSync('results')
    .filter(
      file => file.startsWith('userDataWithRepos_v2') && file.endsWith('.json')
    )
    .map(file =>
      userList.push(...JSON.parse(fs.readFileSync(`results/${file}`, 'utf8')))
    );

  const file = fs.readFileSync(
    'results/github_profiles_total_v4.2.csv',
    'utf8'
  );

  if (file === undefined || file.length === 0) {
    console.error('파일이 없어요!');
    return;
  }
  const lines = file.split('\n');

  const result: CSVInterface[] = [];

  for (const line of lines.slice(1)) {
    const lineData = line.split(',');
    const user = userList.find(user => user.id === Number(lineData[1]));
    if (user) {
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

      for (const repo of user.repos) {
        if (repo.language && isLanguage(repo.language)) {
          userText += repo.name.replace(/,/g, '&') + ' :: ';
          userText += repo.description
            ? repo.description.replace(/,/g, '&') + ' / '
            : ' / ';
          languageCount[repo.language] += 1;
        }
      }
      const totalCount = Object.values(languageCount).reduce(
        (acc, curr) => acc + curr,
        0
      );
      if (totalCount > 0) {
        for (const lang of languages) {
          languageCount[lang] = Number(
            (languageCount[lang] / totalCount).toFixed(3)
          );
        }
      } else {
        continue;
      }

      result.push({
        ...languageCount,
        username: user.login,
        userID: user.id,
        repoCount: user.repos.length,
        text: userText,
        stack: lineData[21],
        note: lineData[22],
      });
    }
  }

  let csvContent = `user_ID, username, repo_count, ${languages.join(
    ', '
  )}, text, stack, note\n`;
  for (const line of result) {
    const row = [
      line.userID,
      line.username,
      line.repoCount,
      ...languages.map(lang => line[lang]),
      line.text,
      line.stack,
      line.note,
    ].join(',');
    csvContent += `${row}\n`;
  }
  fs.writeFileSync('results/github_profiles_total_v4.3.csv', csvContent);
}

main();
