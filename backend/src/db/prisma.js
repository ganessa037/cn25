import { PrismaClient } from '@prisma/client';

const globalForPrisma = globalThis;
export const prisma =
  globalForPrisma.prisma ||
  new PrismaClient({
    log: ['error', 'warn'], // 需要可加 'query'
  });

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma;
}

export default prisma;