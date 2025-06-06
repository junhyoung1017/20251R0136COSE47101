import { useState } from 'react';

import './App.css';
import { getUserData } from './utils/getUserData';
import { getOpenAIAnswer } from './utils/getOpenAIAnswer';
import { getSimilarUsers } from './utils/getSimilarUsers';
import type { AwesomeUserInterface } from './models/models';
import { getCoworkers } from './utils/getCoworkers';

function App() {
  const [githubId, setGithubId] = useState('');
  const [stack, setStack] = useState<string>('');
  const [similarUsers, setSimilarUsers] = useState<AwesomeUserInterface[]>([]);
  const [coworkers, setCoworkers] = useState<AwesomeUserInterface[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    getUserData(githubId).then(data => {
      if ('message' in data) {
        setError(data.message);
        setIsLoading(false);
        return;
      }
      getOpenAIAnswer(data)
        .then(stack => {
          setStack(stack);
          setSimilarUsers(getSimilarUsers(stack));
          setCoworkers(getCoworkers(stack));
          setIsLoading(false);
        })
        .catch(error => {
          setError(error.message);
          setIsLoading(false);
        });
    });
  };

  return (
    <div className="container">
      <div className="branding">고컴에어공군</div>
      <h1 className="title">깃허브 유저 아이디를 입력해주세요</h1>
      <form className="form" onSubmit={onSubmit}>
        <input
          type="text"
          id="github-id"
          className="input"
          placeholder="깃허브 유저 아이디"
          value={githubId}
          onChange={e => {
            setGithubId(e.target.value);
            setSimilarUsers([]);
            setCoworkers([]);
            setStack('');
            setError(null);
          }}
          disabled={isLoading}
        />
        <button type="submit" className="button" disabled={isLoading}>
          {isLoading ? '분석 중...' : '확인'}
        </button>
        {error && <div className="error-message">{error}</div>}
        {stack && (
          <div className="stack-result">
            <span className="username">{githubId}</span>님의 추정 스택은{' '}
            <h1 className="stack">{stack}</h1>
          </div>
        )}
        <h2 className="subtitle">멘토 추천</h2>
        <div className="user-list">
          {similarUsers.map(user => (
            <div key={user.userID} className="user-card mentor">
              <a
                href={`https://github.com/${user.username}`}
                target="_blank"
                rel="noopener noreferrer"
                className="user-link"
              >
                {user.username}
              </a>
            </div>
          ))}
        </div>
        <h2 className="subtitle">코워킹 파트너</h2>
        <div className="user-list">
          {coworkers.map(user => (
            <div key={user.userID} className="user-card coworker">
              <a
                href={`https://github.com/${user.username}`}
                target="_blank"
                rel="noopener noreferrer"
                className="user-link"
              >
                {user.username}
              </a>
              <p className="user-stack">{user.stack} 개발자</p>
            </div>
          ))}
        </div>
      </form>
    </div>
  );
}

export default App;
