export const recommendationMap: Record<string, string[]> = {
  Frontend: ['Server', 'Android', 'iOS'],
  Server: ['Frontend', 'Android', 'iOS'],
  Android: ['Server', 'iOS'],
  iOS: ['Server', 'Android'],
  System: ['Server'],
  'ML-Data': ['Visualization'],
  Visualization: ['Server', 'Frontend'],
};
