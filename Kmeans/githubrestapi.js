// 필요한 모듈 불러오기
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const natural = require('natural');
const sw = require('stopword');
const lda = require('lda');
const { PCA } = require('ml-pca');
const { marked } = require('marked');
const cheerio = require('cheerio');

// 실제 DOMParser 대신 단순 텍스트 정제 함수 사용
function cleanHtmlContent(htmlContent) {
  if (!htmlContent || typeof htmlContent !== 'string') {
    return '';
  }
  
  // HTML 태그 제거
  let text = htmlContent.replace(/<[^>]*>/g, ' ');
  
  // HTML 엔티티 디코딩
  text = text.replace(/&lt;/g, '<')
             .replace(/&gt;/g, '>')
             .replace(/&amp;/g, '&')
             .replace(/&quot;/g, '"')
             .replace(/&#39;/g, "'")
             .replace(/&nbsp;/g, ' ');
  
  // 여러 공백을 단일 공백으로
  text = text.replace(/\s+/g, ' ').trim();
  
  return text;
}

/**
 * 텍스트 전처리 (개선된 버전 - 더 많은 전처리 단계)
 * @param {string} text - 원본 텍스트
 * @returns {Array<string>} - 전처리된 토큰 배열
 */

/* 사용자의 인기 저장소 3개 가져오기
 * @param {string} username - GitHub 사용자명
 * @returns {Promise<Array<Object>>} - 저장소 정보 배열
 */
async function getTopRepositories(username, count = 3) {
  try {
    console.log(`${username}의 인기 저장소 가져오는 중...`);
    
    // 먼저 사용자의 모든 저장소를 가져와서 정렬
    const response = await githubClient.get(`/users/${username}/repos`, {
      params: {
        sort: 'stars',    // 별표 수로 정렬
        direction: 'desc', // 내림차순
        per_page: 100      // 최대 100개 가져오기
      }
    });
    
    // 결과에서 상위 저장소 선택
    const topRepos = response.data
      .slice(0, count)
      .map(repo => ({
        name: repo.name,
        url: repo.html_url,
        description: repo.description || '',
        stars: repo.stargazers_count,
        forks: repo.forks_count,
        languages_url: repo.languages_url
      }));
    
    // 각 저장소의 언어 정보 가져오기
    for (const repo of topRepos) {
      try {
        const languagesResponse = await githubClient.get(repo.languages_url);
        repo.languages = languagesResponse.data;
        
        // 언어 비율 계산
        const totalBytes = Object.values(repo.languages).reduce((sum, bytes) => sum + bytes, 0);
        repo.languageRatios = {};
        
        for (const [language, bytes] of Object.entries(repo.languages)) {
          repo.languageRatios[language] = parseFloat(((bytes / totalBytes) * 100).toFixed(2));
        }
        
        // API 속도 제한을 방지하기 위한 지연
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error) {
        console.error(`${username}의 저장소 ${repo.name} 언어 정보를 가져오는 중 오류 발생:`, error.message);
        repo.languages = {};
        repo.languageRatios = {};
      }
    }
    
    return topRepos;
  } catch (error) {
    console.error(`${username}의 인기 저장소를 가져오는 중 오류 발생:`, error.message);
    return [];
  }
}
// 수집할 언어 목록
const TARGET_LANGUAGES = [
  'Assembly', 'C', 'C++', 'C#', 'Dart', 'Go', 'Java', 'JavaScript', 'Kotlin',
  'MATLAB', 'PHP', 'Python', 'Ruby', 'Rust', 'Scala', 'Swift', 'TypeScript'
];

/**
/**
 * 사용자의 언어 특성 벡터 생성
 * @param {Array<Object>} repositories - 저장소 정보 배열
 * @returns {Object} - 언어 특성 벡터
 */
function createLanguageFeatureVector(repositories) {
  // 모든 저장소의 언어 비율을 결합
  const languageCounts = {};
  let totalRepos = repositories.length;
  
  if (totalRepos === 0) {
    return {};
  }
  
  repositories.forEach((repo) => {
    const weight = 1 + Math.log(repo.stars + 1) / 10;
    Object.entries(repo.languageRatios || {}).forEach(([language, percentage]) => {
      // 원하는 언어만 누적
      if (TARGET_LANGUAGES.includes(language)) {
        languageCounts[language] = (languageCounts[language] || 0) + (percentage * weight);
      }
    });
  });

  const totalCounts = Object.values(languageCounts).reduce((sum, count) => sum + count, 0);

  const languageFeatures = {};
  if (totalCounts > 0) {
    TARGET_LANGUAGES.forEach(language => {
      languageFeatures[language] = languageCounts[language]
        ? parseFloat((languageCounts[language] / totalCounts).toFixed(4))
        : 0;
    });
  } else {
    // 모든 값 0으로 초기화
    TARGET_LANGUAGES.forEach(language => {
      languageFeatures[language] = 0;
    });
  }

  return languageFeatures;
}
/**
 * 특정 ID 범위의 사용자 정보 가져오기
 * @param {number} startId - 시작 사용자 ID
 * @param {number} count - 가져올 사용자 수
 * @returns {Promise<Array<{id: number, username: string}>>} - 사용자 정보 배열
 */
/**
 * 특정 ID부터 minRepos 이상인 사용자 count명 수집
 * @param {number} startId - 시작 사용자 ID
 * @param {number} count - 수집할 사용자 수
 * @param {number} minRepos - 최소 저장소 수
 * @returns {Promise<Array<{id: number, username: string, repos: number}>>}
 */
async function getActiveUsersById(startId, count, minRepos = 3) {
  const users = [];
  let userId = startId;
  let consecutiveErrors = 0;
  const maxConsecutiveErrors = 10;

  console.log(`ID ${startId}부터 저장소 ${minRepos}개 이상인 사용자 ${count}명을 수집합니다...`);

  while (users.length < count) {
    try {
      const response = await githubClient.get(`/user/${userId}`);
      if (response.status === 200 && response.data && response.data.login) {
        // 저장소 개수 확인
        const userInfo = await githubClient.get(`/users/${response.data.login}`);
        const publicRepos = userInfo.data.public_repos || 0;
        if (publicRepos >= minRepos) {
          users.push({
            id: userId,
            username: response.data.login,
            repos: publicRepos
          });
          console.log(`ID ${userId} (${response.data.login}) 추가됨 (저장소 ${publicRepos}개)`);
        } else {
          console.log(`ID ${userId} (${response.data.login}) 저장소 부족 (${publicRepos}개)`);
        }
        consecutiveErrors = 0;
      }} catch (error) {
      consecutiveErrors++;
      if (error.response && error.response.status === 404) {
        console.warn(`ID ${userId}의 사용자를 찾을 수 없습니다.`);
      } else if (error.response && error.response.status === 403) {
        console.error('GitHub API 속도 제한에 도달했습니다. 1시간 후에 다시 시도하세요.');
        break;
      } else {
        console.error(`ID ${userId}의 사용자 정보를 가져오는 중 오류 발생:`, error.message);
      }
      if (consecutiveErrors >= maxConsecutiveErrors) {
        console.error(`연속 ${maxConsecutiveErrors}번의 오류가 발생하여 사용자 수집을 중단합니다.`);
        break;
      }
    }
    userId++;
    // API 속도 제한 방지
    if (users.length % 10 === 0 && users.length > 0) {
      console.log('API 속도 제한 방지를 위해 10초 대기 중...');
      await new Promise(resolve => setTimeout(resolve, 10000));
    } else {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  console.log(`총 ${users.length}명의 활성 사용자를 수집했습니다.`);
  return users;
}
// .env 파일 로드
dotenv.config();

// GitHub API 토큰 설정
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;

// GitHub API를 위한 클라이언트 설정
const githubClient = axios.create({
  baseURL: 'https://api.github.com',
  headers: {
    'Authorization': `Bearer ${GITHUB_TOKEN}`,
    'Accept': 'application/vnd.github.v3+json'
  },
  // 요청 타임아웃 설정
  timeout: 10000
});

/**
 * 사용자의 README 가져오기 (HTML 파싱 오류 수정 버전)
 * @param {string} username - GitHub 사용자명
 * @returns {Promise<string>} - README 내용
 */
async function getUserReadme(username) {
  try {
    console.log(`${username}의 README를 가져오는 중...`);
    
    // 1. 먼저 사용자의 메인 저장소 (username/username) README 시도
    try {
      const response = await githubClient.get(`/repos/${username}/${username}/readme`, {
        headers: {
          'Accept': 'application/vnd.github.raw'
        }
      });
      
      console.log(`${username}의 프로필 README를 성공적으로 가져왔습니다.`);
      return response.data;
    } catch (profileError) {
      console.log(`${username}의 프로필 README를 찾을 수 없습니다. 다른 방법을 시도합니다.`);
      
      // 2. 사용자의 프로필 정보 시도
      try {
        const profileResponse = await githubClient.get(`/users/${username}`);
        const bio = profileResponse.data.bio || '';
        
        if (bio && bio.length > 0) {
          console.log(`${username}의 프로필 바이오를 가져왔습니다.`);
          return bio;
        } else {
          console.log(`${username}의 프로필 바이오가 비어 있습니다. 다른 방법을 시도합니다.`);
        }
      } catch (bioError) {
        console.log(`${username}의 프로필 정보를 가져오는데 실패했습니다. 다른 방법을 시도합니다.`);
      }
      
      // 3. 사용자의 인기 저장소의 README 시도
      try {
        const reposResponse = await githubClient.get(`/users/${username}/repos`, {
          params: {
            sort: 'stars',
            direction: 'desc',
            per_page: 3
          }
        });
        
        if (reposResponse.data && reposResponse.data.length > 0) {
          const topRepo = reposResponse.data[0];
          
          try {
            const readmeResponse = await githubClient.get(`/repos/${username}/${topRepo.name}/readme`, {
              headers: {
                'Accept': 'application/vnd.github.raw'
              }
            });
            
            console.log(`${username}의 인기 저장소 ${topRepo.name}에서 README를 가져왔습니다.`);
            return readmeResponse.data;
          } catch (readmeError) {
            console.log(`${username}의 인기 저장소 README를 가져오는데 실패했습니다.`);
          }
        }
      } catch (reposError) {
        console.log(`${username}의 저장소 목록을 가져오는데 실패했습니다.`);
      }
    }
    
    // 4. 마지막 대안으로 사용자의 퍼블릭 활동 내용 가져오기
    try {
      const eventsResponse = await githubClient.get(`/users/${username}/events/public`, {
        params: {
          per_page: 10
        }
      });
      
      if (eventsResponse.data && eventsResponse.data.length > 0) {
        // 이벤트에서 텍스트 추출 (커밋 메시지, 이슈 코멘트 등)
        const eventTexts = eventsResponse.data
          .map(event => {
            if (event.type === 'IssueCommentEvent' && event.payload && event.payload.comment) {
              return event.payload.comment.body || '';
            }
            if (event.type === 'PushEvent' && event.payload && event.payload.commits) {
              return event.payload.commits.map(commit => commit.message || '').join(' ');
            }
            if (event.type === 'CreateEvent' && event.payload && event.payload.description) {
              return event.payload.description || '';
            }
            return '';
          })
          .filter(text => text.length > 0)
          .join(' ');
        
        if (eventTexts.length > 0) {
          console.log(`${username}의 퍼블릭 활동 내용을 가져왔습니다.`);
          return eventTexts;
        }
      }
    } catch (eventsError) {
      console.log(`${username}의 퍼블릭 활동을 가져오는데 실패했습니다.`);
    }
    
    // 모든 방법이 실패한 경우 기본 텍스트 반환
    console.warn(`${username}의 README를 가져오는 모든 방법이 실패했습니다. 기본 텍스트를 사용합니다.`);
    return `GitHub user ${username}. Programming. Development. Code. Software. Repository.`;
    
  } catch (error) {
    console.error(`${username}의 README를 가져오는 중 오류 발생:`, error.message);
    // 오류 발생시 기본 텍스트 반환
    return `GitHub user ${username}. Programming. Development. Code. Software. Repository.`;
  }
}

/**
 * 사용자의 저장소 설명 가져오기 (개선된 버전 - 오류 처리 강화)
 * @param {string} username - GitHub 사용자명
 * @param {number} repoCount - 가져올 저장소 수 (기본값: 5)
 * @returns {Promise<string>} - 결합된 저장소 설명
 */
async function getRepositoryDescriptions(username, repoCount = 5) {
  try {
    console.log(`${username}의 저장소 설명을 가져오는 중...`);
    
    const response = await githubClient.get(`/users/${username}/repos`, {
      params: {
        sort: 'updated',
        direction: 'desc',
        per_page: repoCount
      }
    });
    
    const descriptions = response.data
      .map(repo => {
        // 저장소 이름, 설명, 토픽 등을 포함
        const description = repo.description || '';
        const topics = repo.topics && Array.isArray(repo.topics) ? repo.topics.join(' ') : '';
        const name = repo.name || '';
        
        return `${name} ${description} ${topics}`.trim();
      })
      .filter(desc => desc.trim().length > 0);
    
    if (descriptions.length > 0) {
      console.log(`${username}의 저장소 설명 ${descriptions.length}개를 성공적으로 가져왔습니다.`);
      return descriptions.join(' ');
    }
    
    // 저장소 설명이 없는 경우 저장소 이름이라도 사용
    const repoNames = response.data
      .map(repo => repo.name || '')
      .filter(name => name.trim().length > 0);
    
    if (repoNames.length > 0) {
      console.log(`${username}의 저장소 설명이 없어서 저장소 이름 ${repoNames.length}개를 사용합니다.`);
      return repoNames.join(' ');
    }
    
    console.warn(`${username}의 저장소 정보가 없습니다.`);
    return '';
  } catch (error) {
    console.error(`${username}의 저장소 설명을 가져오는 중 오류 발생:`, error.message);
    
    // 오류가 발생해도 빈 문자열 대신 기본 설명 반환
    return `repositories projects code software development ${username}`;
  }
}

/**
 * 마크다운에서 코드 블록 제거
 * @param {string} markdown - 마크다운 텍스트
 * @returns {string} - 코드 블록이 제거된 텍스트
 */
function removeCodeBlocks(markdown) {
  // ```로 둘러싸인 코드 블록 제거
  let text = markdown.replace(/```[\s\S]*?```/g, '');
  
  // 인라인 코드 제거 (`로 둘러싸인 텍스트)
  text = text.replace(/`[^`]*`/g, '');
  
  return text;
}

/**
 * 마크다운에서 URL 제거
 * @param {string} markdown - 마크다운 텍스트
 * @returns {string} - URL이 제거된 텍스트
 */
function removeUrls(markdown) {
  // [텍스트](URL) 형식의 링크에서 텍스트만 유지
  let text = markdown.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
  
  // 일반 URL 제거
  text = text.replace(/https?:\/\/\S+/g, '');
  
  return text;
}

/**
 * 마크다운에서 HTML 태그 제거
 * @param {string} markdown - 마크다운 텍스트
 * @returns {string} - HTML 태그가 제거된 텍스트
 */
function removeHtmlTags(markdown) {
  return markdown.replace(/<[^>]*>/g, '');
}

/**
 * 텍스트 전처리
 * @param {string} text - 원본 텍스트
 * @returns {Array<string>} - 전처리된 토큰 배열
 */
// 개선된 마크다운 및 특수 문자 제거 함수 - 더 강력한 버전
function removeSpecialCharacters(text) {
  if (!text || typeof text !== 'string') {
    return '';
  }
  
  try {
    // 이스케이프된 문자 해석 (\n, \t, \", \\, 등)
    let cleanText = text
      .replace(/\\n/g, '\n')
      .replace(/\\r/g, '\r')
      .replace(/\\t/g, ' ')
      .replace(/\\"/g, '"')
      .replace(/\\\\/g, '\\');

    // HTML 엔티티 디코딩
    cleanText = cleanText.replace(/&lt;/g, '<')
                         .replace(/&gt;/g, '>')
                         .replace(/&amp;/g, '&')
                         .replace(/&quot;/g, '"')
                         .replace(/&#39;/g, "'")
                         .replace(/&nbsp;/g, ' ');
    cleanText = text.replace(/```[\s\S]*?```/g, ' ');
        // HTML 주석 제거
    cleanText = cleanText.replace(/<!--[\s\S]*?-->/g, ' ');

    // 이미지 마크다운 (줄바꿈 허용 포함) 제거
    cleanText = cleanText.replace(/!\[[^\]]*\]\([^\)]*\)/gm, ' ');
    // 마크다운 인라인 코드 제거
    cleanText = cleanText.replace(/`[^`]*`/g, ' ');
    
    // 마크다운 링크 제거
    cleanText = cleanText.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    
    // 마크다운 이미지 제거
    cleanText = cleanText.replace(/!\[.*?\]\(.*?\)/g, ' ');
    
    // HTML 태그 제거
    cleanText = cleanText.replace(/<[^>]*>/g, ' ');
    
    // URL 제거
    cleanText = cleanText.replace(/https?:\/\/\S+/g, ' ');
    
    // 마크다운 헤더 기호(#) 제거
    cleanText = cleanText.replace(/^#+\s+/gm, ' ');
    
    // 마크다운 기호 제거(*, -, >, =, +, |, \, /, $, ., :, ;, (, ), [, ], {, }, !, ?)
    cleanText = cleanText.replace(/[*\->=+|\\/$.,;:()\[\]{}\?!#]/g, ' ');
    
    // 줄바꿈 및 탭을 공백으로 대체
    cleanText = cleanText.replace(/[\n\r\t]/g, ' ');
    
    // 이모지 및 유니코드 특수 문자 제거
    cleanText = cleanText.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}]/gu, ' ');
    
    // 여러 개의 공백을 하나의 공백으로 대체
    cleanText = cleanText.replace(/\s+/g, ' ').trim();
    
    return cleanText;
  } catch (error) {
    console.error('특수 문자 제거 중 오류 발생:', error.message);
    return text; // 오류 발생 시 원본 텍스트 반환
  }
}

// 개선된 텍스트 전처리 함수
/*function improvedPreprocessText(text) {
  if (!text || typeof text !== 'string' || text.trim().length === 0) {
    console.warn('전처리할 텍스트가 없습니다. 기본 토큰을 반환합니다.');
    return ['github', 'user', 'repository', 'code', 'software', 'development', 'programming'];
  }

  try {
    // 특수 문자 및 마크다운 제거
    let cleanText = removeSpecialCharacters(text);
    
    // 소문자 변환
    cleanText = cleanText.toLowerCase();
    
    // 알파벳, 숫자, 공백 외 문자 제거 (더 엄격하게)
    cleanText = cleanText.replace(/[^a-z0-9\s]/g, ' ');
    
    // 다중 공백 제거
    cleanText = cleanText.replace(/\s+/g, ' ').trim();
    
    if (cleanText.length === 0) {
      console.warn('텍스트 정제 후 내용이 없습니다. 기본 토큰을 반환합니다.');
      return ['github', 'user', 'repository', 'code', 'software', 'development', 'programming'];
    }
    
    // 토큰화
    const tokens = cleanText.split(/\s+/);
    
    // 토큰 길이 필터링 (2글자 이하 제거, 20글자 초과 제거)
    const filteredByLength = tokens.filter(token => token.length > 2 && token.length < 20);
    
    // 불용어 제거 (영어 불용어)
    const filteredTokens = sw.removeStopwords(filteredByLength);
    
    // 어간 추출 (Stemming)
    const stemmer = natural.PorterStemmer;
    const stemmedTokens = filteredTokens.map(token => stemmer.stem(token));
    
    // 중복 토큰 제거
    const uniqueTokens = [...new Set(stemmedTokens)];
    
    // 최소 개수 확인
    if (uniqueTokens.length < 5) {
      console.warn('전처리 후 토큰이 너무 적습니다. 기본 토큰을 추가합니다.');
      uniqueTokens.push('github', 'user', 'repository', 'code', 'software', 'development', 'programming');
    }
    
    return uniqueTokens;
  } catch (error) {
    console.error('텍스트 전처리 중 오류 발생:', error.message);
    return ['github', 'user', 'repository', 'code', 'software', 'development', 'programming'];
  }
}*/
/**
 * 텍스트 전처리 (완전히 개선된 버전)
 * @param {string} text - 원본 텍스트
 * @returns {Array<string>} - 전처리된 토큰 배열
 */
function improvedPreprocessText(text) {
  // 기본 토큰 (텍스트가 없거나 처리 실패 시 사용)
  const defaultTokens = ['github', 'user', 'repository', 'code', 'software', 'development', 'programming'];
  
  if (!text || typeof text !== 'string' || text.trim().length === 0) {
    console.warn('전처리할 텍스트가 없습니다. 기본 토큰을 반환합니다.');
    return defaultTokens;
  }

  try {
    // 마크다운/HTML에서 텍스트 추출
    let cleanText = extractPlainTextFromMarkdown(text);
    
    // 소문자 변환
    cleanText = cleanText.toLowerCase();
    
    // 알파벳, 숫자, 공백 외 제거 (더 엄격하게)
    cleanText = cleanText.replace(/[^a-z0-9\s]/g, ' ');
    
    // 다중 공백 제거
    cleanText = cleanText.replace(/\s+/g, ' ').trim();
    
    if (cleanText.length === 0) {
      console.warn('텍스트 정제 후 내용이 없습니다. 기본 토큰을 반환합니다.');
      return defaultTokens;
    }
    
    // 토큰화
    const tokens = cleanText.split(/\s+/);
    
    // 토큰 길이 필터링 (2글자 이하 제거, 20글자 초과 제거)
    const filteredByLength = tokens.filter(token => token.length > 2 && token.length < 20);
    
    // 불용어 제거 (영어 불용어)
    const filteredTokens = sw.removeStopwords(filteredByLength);
    
    // 어간 추출 (Stemming)
    const stemmer = natural.PorterStemmer;
    const stemmedTokens = filteredTokens.map(token => stemmer.stem(token));
    
    // 중복 토큰 제거
    const uniqueTokens = [...new Set(stemmedTokens)];
    
    // 최소 개수 확인
    if (uniqueTokens.length < 5) {
      console.warn('전처리 후 토큰이 너무 적습니다. 기본 토큰을 추가합니다.');
      return [...uniqueTokens, ...defaultTokens.filter(token => !uniqueTokens.includes(token))];
    }
    
    return uniqueTokens;
  } catch (error) {
    console.error('텍스트 전처리 중 오류 발생:', error.message);
    return defaultTokens;
  }
}

/**
 * 마크다운 및 HTML 전처리 함수 (marked + cheerio 기반)
 * @param {string} markdown - 마크다운 또는 HTML이 섞인 텍스트
 * @returns {string} - 정제된 평문 텍스트
 */
function extractPlainTextFromMarkdown(markdown) {
  if (!markdown || typeof markdown !== 'string') return '';

  try {
    // 마크다운을 HTML로 변환
    const html = marked.parse(markdown);

    // cheerio로 HTML 파싱
    const $ = cheerio.load(html);

    // 코드 블록, 스크립트, 스타일, 이미지 등 제거
    $('pre, code, script, style, img, svg').remove();

    // 텍스트 추출
    let text = $.text();

    // HTML 태그 제거 (혹시 남아있을 경우)
    text = text.replace(/<[^>]*>/g, ' ');

    // URL 제거 (http, https, www 등)
    text = text.replace(/https?:\/\/\S+/g, ' ');
    text = text.replace(/www\.\S+/g, ' ');

    // 여러 공백을 하나로
    text = text.replace(/\s+/g, ' ').trim();

    return text;
  } catch (err) {
    console.error('마크다운 정제 오류:', err.message);

    // 오류 발생 시 간단한 텍스트 정제 시도
    if (markdown) {
      return markdown
        .replace(/```[\s\S]*?```/g, ' ')    // 코드 블록 제거
        .replace(/`[^`]*`/g, ' ')           // 인라인 코드 제거
        .replace(/<[^>]*>/g, ' ')           // HTML 태그 제거
        .replace(/https?:\/\/\S+/g, ' ')    // URL 제거
        .replace(/www\.\S+/g, ' ')          // URL 제거
        .replace(/\s+/g, ' ')               // 여러 공백을 하나로
        .trim();
    }
    return '';
  }
}

/**
 * TF-IDF 벡터 생성
 * @param {Array<{username: string, tokens: Array<string>}>} userTokens - 사용자별 토큰 배열
 * @returns {Object} - TF-IDF 결과 객체
 */
function createTfidfVectors(userTokens) {
  // TfIdf 인스턴스 생성
  const tfidf = new natural.TfIdf();
  
  // 각 사용자 문서를 TF-IDF에 추가
  userTokens.forEach(user => {
    tfidf.addDocument(user.tokens);
  });
  
  // 모든 용어 수집
  const terms = new Set();
  userTokens.forEach(user => {
    user.tokens.forEach(token => {
      terms.add(token);
    });
  });
  
  // 각 사용자의 TF-IDF 벡터 생성
  const userVectors = [];
  const allTerms = Array.from(terms);
  
  for (let i = 0; i < userTokens.length; i++) {
    const username = userTokens[i].username;
    const vector = [];
    
    allTerms.forEach(term => {
      const tfidfValue = tfidf.tfidf(term, i);
      vector.push(tfidfValue);
    });
    
    userVectors.push({
      username,
      vector
    });
  }
  
  return {
    terms: allTerms,
    userVectors
  };
}

/*
 * LDA 토픽 모델링 (오류 처리 강화)
 * @param {Array<{username: string, tokens: Array<string>}>} userTokens - 사용자별 토큰 배열
 * @param {number} numTopics - 토픽 수
 * @param {number} numTerms - 각 토픽별 용어 수
 * @returns {Object} - LDA 결과 객체
 */
/*function improvedPerformLDA(userTokens, numTopics = 3, numTerms = 100) {
  try {
    console.log('LDA 모델링 시작...');
    
   // 사용자별 토큰이 적어도 한 개 이상 있는지 확인
    const validUserTokens = userTokens.filter(user =>
      user.tokens && Array.isArray(user.tokens) && user.tokens.length > 0
    );

    if (validUserTokens.length === 0) {
      console.error('유효한 사용자 토큰이 없습니다.');
      return {
        topicTerms: [],
        userTopicMatrix: []
      };
    }

    // 문서를 문자열 배열로 변환 (각 토큰을 공백으로 구분)
    const documents = validUserTokens.map(user => user.tokens.join(' ')); // 1차원 배열로 변경

    console.log(`${validUserTokens.length}명의 사용자에 대해 LDA 수행...`);

    // LDA 수행
    const ldaResults = lda(documents, numTopics, numTerms);
    
    // 결과 유효성 검사
    if (!ldaResults || !Array.isArray(ldaResults) || ldaResults.length === 0) {
      throw new Error('LDA가 유효한 결과를 반환하지 않았습니다.');
    }
    
    console.log('LDA 결과 생성 완료, 사용자별 토픽 분포 계산 중...');
    
    // 사용자별 토픽 분포 계산
    const userTopicMatrix = [];
    
    for (let userIndex = 0; userIndex < validUserTokens.length; userIndex++) {
      const username = validUserTokens[userIndex].username;
      const userTokensStr = validUserTokens[userIndex].tokens.join(' ');
      const topics = {};
      
      // 각 토픽에 대한 사용자 문서의 관련성 계산
      ldaResults.forEach((topic, topicIndex) => {
        let topicScore = 0;
        
        topic.forEach(termObj => {
          const term = termObj.term;
          const probability = termObj.probability;
          
          // 문서에 용어가 있으면 확률을 점수에 추가
          if (userTokensStr.includes(term)) {
            topicScore += probability;
          }
        });
        
        topics[`topic_${topicIndex}`] = topicScore;
      });
      
      // 토픽 점수 정규화 (합이 1이 되도록)
      const totalScore = Object.values(topics).reduce((sum, score) => sum + score, 0);
      if (totalScore > 0) {
        Object.keys(topics).forEach(topic => {
          topics[topic] /= totalScore;
        });
      }
      
      userTopicMatrix.push({
        username,
        topics
      });
    }
    
    console.log('LDA 분석 완료!');
    
    return {
      topicTerms: ldaResults,
      userTopicMatrix
    };
  } catch (error) {
    console.error('LDA 수행 중 오류 발생:', error.message);
    console.error('오류 세부 정보:', error.stack);
    
    // 오류 발생 시 빈 결과 반환
    return {
      topicTerms: [],
      userTopicMatrix: []
    };
  }
}
*/
/**
 * 차원 축소 (PCA)
 * @param {Array<{username: string, vector: Array<number>}>} vectors - 벡터 배열
 * @param {number} dimensions - 축소할 차원 수
 * @returns {Array<{username: string, vector: Array<number>}>} - 축소된 벡터
 */
function reduceDimensions(vectors, dimensions = 2) {
  // 벡터 데이터만 추출
  const data = vectors.map(item => item.vector);
  
  // PCA 수행
  const pca = new PCA(data);
  const reducedData = pca.predict(data, { nComponents: dimensions });
  
  // 결과 포맷팅
  return vectors.map((item, index) => ({
    username: item.username,
    vector: Array.from(reducedData.getRow(index))
  }));
}

/**
 * 데이터 저장
 * @param {string} filePath - 저장할 파일 경로
 * @param {Object} data - 저장할 데이터
 */
function saveData(filePath, data) {
  try {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    console.log(`데이터가 ${filePath}에 저장되었습니다.`);
  } catch (error) {
    console.error(`데이터 저장 중 오류 발생:`, error.message);
  }
}

/**
 * 사용자 README 분석 메인 함수
 * @param {Array<string>} usernames - GitHub 사용자명 배열
 * @param {string} outputDir - 결과를 저장할 디렉토리 경로
 */
async function analyzeUserReadmes(usernames, outputDir) {
  // 결과 디렉토리 생성
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  const userReadmes = [];
  const userTokens = [];
  const userRepositories = [];
  const userLanguageFeatures = [];
  
  // 각 사용자의 README 및 저장소 정보 수집
  for (const username of usernames) {
    console.log(`\n${username}의 정보 분석 중...`);
    
    // README 가져오기
    const readme = await getUserReadme(username);
    
    // 저장소 설명 가져오기
    const repoDescriptions = await getRepositoryDescriptions(username);
    
    // README와 저장소 설명 결합
    const combinedText = `${readme} ${repoDescriptions}`;
    console.log(`[${username}] 전처리 전 텍스트 예시:`, combinedText.slice(0, 500));
    userReadmes.push({
      username,
      text: combinedText
    });
    
    // 텍스트 전처리
    const tokens = improvedPreprocessText(combinedText);
    // 전처리 후 토큰 일부 확인
    console.log(`[${username}] 전처리 후 토큰 예시:`, tokens.slice(0, 20));
    // 토큰이 너무 적으면 기본 토큰 추가 (LDA 실패 방지)
    if (tokens.length < 10) {
      tokens.push('github', 'user', 'profile', 'code', 'develop', 'project', 'software', 'repository', 'programming', 'developer');
    }
    
    userTokens.push({
      username,
      tokens
    });
    
    console.log(`${username}의 토큰 수: ${tokens.length}`);
    
    // 인기 저장소 3개 가져오기
    const topRepos = await getTopRepositories(username, 3);
    
    userRepositories.push({
      username,
      repositories: topRepos
    });
    
    // 언어 특성 벡터 생성
    const languageFeatures = createLanguageFeatureVector(topRepos);
    
    userLanguageFeatures.push({
      username,
      language_features: languageFeatures
    });
    
    console.log(`${username}의 언어 특성:`, Object.keys(languageFeatures).join(', '));
  }
  
  // 결과 저장
  saveData(path.join(outputDir, 'user_readmes.json'), userReadmes);
  saveData(path.join(outputDir, 'user_tokens.json'), userTokens);
  saveData(path.join(outputDir, 'user_repositories.json'), userRepositories);
  saveData(path.join(outputDir, 'user_language_features.json'), userLanguageFeatures);
  
  // TF-IDF 벡터 생성
  console.log('\nTF-IDF 벡터 생성 중...');
  const tfidfResults = createTfidfVectors(userTokens);
  saveData(path.join(outputDir, 'tfidf_vectors.json'), tfidfResults);
  
  // TF-IDF 벡터 차원 축소 (시각화용)
  if (userTokens.length >= 3) {  // PCA는 최소 3개 이상의 데이터 필요
    console.log('TF-IDF 벡터 차원 축소 중...');
    const reducedTfidf = reduceDimensions(tfidfResults.userVectors, 2);
    saveData(path.join(outputDir, 'tfidf_reduced.json'), reducedTfidf);
  }
  
  /*// LDA 토픽 모델링
  let ldaResults = null;
  try {
    console.log('\nLDA 토픽 모델링 수행 중...');
    ldaResults = improvedPerformLDA(userTokens); // <<<<< 할당
    saveData(path.join(outputDir, 'lda_results.json'), ldaResults);
  } catch (error) {
    console.error('LDA 토픽 모델링 중 오류 발생:', error.message);
    console.log('LDA 결과 없이 계속 진행합니다.');
  }*/
  
   // 종합 사용자 프로필 생성
  const userProfiles = usernames.map(username => {
    // README 텍스트 특성
    const readmeFeatures = tfidfResults.userVectors.find(u => u.username === username);

    // 언어 특성
    const languageFeature = userLanguageFeatures.find(u => u.username === username);

    /* 토픽 특성 (LDA가 성공한 경우)
    let topicFeatures = {};
    try {
      if (ldaResults && ldaResults.userTopicMatrix) {
        const topics = ldaResults.userTopicMatrix.find(u => u.username === username);
        if (topics) {
          topicFeatures = topics.topics;
        }
      }
    } catch (error) {
      console.warn(`${username}의 토픽 특성을 가져오는 중 오류 발생:`, error.message);
    }*/

    return {
      username,
      language_features: languageFeature ? languageFeature.language_features : {},
      //topic_features: topicFeatures
    };
  });
  saveData(path.join(outputDir, 'user_profiles.json'), userProfiles);
  
  console.log('\n분석이 완료되었습니다!');
  
  return {
    userReadmes,
    userTokens,
    userRepositories,
    userLanguageFeatures,
    userProfiles
  };
}


async function main() {
  try {
    // 결과 저장 디렉토리
    const outputDir = 'C:\\Users\\jun01\\OneDrive\\바탕 화면\\고려대\\데과\\TermProject\\github\\nlp_results';
    
    // 명령행 인자 파싱
    const args = process.argv.slice(2);
    let mode = 'username';
    let startId = 1;
    let count = 10;
    let usernames = [];
    
    for (let i = 0; i < args.length; i++) {
      if (args[i] === '--id-range') {
        mode = 'id-range';
        startId = parseInt(args[i + 1]) || 1;
        count = parseInt(args[i + 2]) || 10;
        i += 2;
      } else if (args[i] === '--username') {
        mode = 'username';
        while (i + 1 < args.length && !args[i + 1].startsWith('--')) {
          usernames.push(args[i + 1]);
          i++;
        }
      }
    }
    
    // 사용자 정보 수집
    let users = [];
    
    if (mode === 'id-range') {
      console.log(`ID 범위 모드: ${startId}부터 저장소 3개 이상인 사용자 ${count}명을 수집합니다.`);
  // 활성 사용자만 바로 수집 (최소 3개의 저장소를 가진 사용자)
      users = await getActiveUsersById(startId, count, 3);

      // 사용자 정보 저장
      saveData(path.join(outputDir, 'users_info.json'), users);

      // 사용자명 추출
      usernames = users.map(user => user.username);
    } else {
      // 커맨드 라인 인자가 없으면 기본 사용자 목록 사용
      if (usernames.length === 0) {
        console.log('GitHub 사용자명이 지정되지 않았습니다. 기본 사용자 목록을 사용합니다.');
        usernames = ['octocat', 'torvalds', 'gaearon', 'yyx990803', 'tj'];
      }
      
      console.log(`사용자명 모드: ${usernames.join(', ')} 사용자의 정보를 분석합니다.`);
    }
    
    // README 분석 수행
    await analyzeUserReadmes(usernames, outputDir);
  } catch (error) {
    console.error('메인 함수 실행 중 오류 발생:', error.message);
  }
}

// 스크립트 실행
main();