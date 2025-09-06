import express from 'express';
import { prisma } from '../db/prisma.js';
import { requireUser } from '../middleware/requireUser.js';

const r = express.Router();
r.use(requireUser);

// list
r.get('/', async (req, res) => {
  const rows = await prisma.vehicle.findMany({ where: { userId: req.user.id }, orderBy: { createdAt: 'desc' }});
  res.json(rows);
});

// create
r.post('/', async (req, res) => {
  const data = req.body;
  const v = await prisma.vehicle.create({
    data: { ...data, userId: req.user.id }
  });
  res.status(201).json(v);
});

// update
r.put('/:id', async (req, res) => {
  const { id } = req.params;
  const data = req.body;
  const v = await prisma.vehicle.update({
    where: { id },
    data
  });
  res.json(v);
});

// delete
r.delete('/:id', async (req, res) => {
  const { id } = req.params;
  await prisma.vehicle.delete({ where: { id }});
  res.json({ success: true });
});

export default r;