import express from 'express';
import { prisma } from '../db/prisma.js';
import { requireUser } from '../middleware/requireUser.js';

const r = express.Router();
r.use(requireUser);

r.get('/', async (req, res) => {
  const rows = await prisma.expense.findMany({
    where: { userId: req.user.id },
    orderBy: { date: 'desc' }
  });
  res.json(rows);
});

r.post('/', async (req, res) => {
  const e = await prisma.expense.create({
    data: { ...req.body, userId: req.user.id }
  });
  res.status(201).json(e);
});

r.put('/:id', async (req, res) => {
  const e = await prisma.expense.update({
    where: { id: req.params.id },
    data: req.body
  });
  res.json(e);
});

r.delete('/:id', async (req, res) => {
  await prisma.expense.delete({ where: { id: req.params.id }});
  res.json({ success: true });
});

export default r;