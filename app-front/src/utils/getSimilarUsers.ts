import type { AwesomeUserInterface } from '../models/models';
import awesomeUsers from '../data/awesome_users.json';

export const getSimilarUsers = (stack: string): AwesomeUserInterface[] => {
  const similarUsers = awesomeUsers.filter(user => user.stack === stack);

  const shuffled = [...similarUsers].sort(() => 0.5 - Math.random());

  return shuffled.slice(0, 3);
};
