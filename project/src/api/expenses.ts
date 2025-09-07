import { api } from "./index";
import type { Expense } from "../types";

export async function listExpenses(params?: { vehicleId?: string }): Promise<Expense[]> {
  const { data } = await api.get<Expense[]>("/expenses", { params });
  return data ?? [];
}

export async function createExpense(payload: Partial<Expense>): Promise<Expense> {
  const { data } = await api.post<Expense>("/expenses", payload);
  return data;
}

export async function updateExpense(id: string, patch: Partial<Expense>): Promise<Expense> {
  const { data } = await api.put<Expense>(`/expenses/${id}`, patch);
  return data;
}

export async function deleteExpense(id: string): Promise<void> {
  await api.delete(`/expenses/${id}`);
}