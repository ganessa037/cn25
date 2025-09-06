import express from 'express';
import { prisma } from '../db/prisma.js';
import { requireUser } from '../middleware/requireUser.js';

const r = express.Router();
r.use(requireUser);

r.get('/', async (req, res) => {
  const rows = await prisma.document.findMany({
    where: { userId: req.user.id },
    orderBy: { uploadedAt: 'desc' }
  });
  res.json(rows);
});

r.post('/', async (req, res) => {
  const d = await prisma.document.create({
    data: { ...req.body, userId: req.user.id }
  });
  res.status(201).json(d);
});

r.put('/:id', async (req, res) => {
  const d = await prisma.document.update({
    where: { id: req.params.id },
    data: req.body
  });
  res.json(d);
});

r.delete('/:id', async (req, res) => {
  await prisma.document.delete({ where: { id: req.params.id }});
  res.json({ success: true });
});

export default r;