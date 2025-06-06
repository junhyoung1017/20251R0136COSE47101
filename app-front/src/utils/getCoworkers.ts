import awesomeUsers from '../data/awesome_users.json';
import { recommendationMap } from '../data/recommendationMap';

export const getCoworkers = (stack: string) => {
  const recommendation = recommendationMap[stack];
  const coworkers = awesomeUsers.filter(user => {
    if (user.stack === stack) return false;
    return recommendation.includes(user.stack);
  });
  const shuffled = [...coworkers].sort(() => 0.5 - Math.random());

  return shuffled.slice(0, 5);
};
