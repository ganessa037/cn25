import { api } from "./index";
import type { Vehicle } from "../types";

export async function listVehicles(): Promise<Vehicle[]> {
  const { data } = await api.get<Vehicle[]>("/vehicles");
  return data ?? [];
}

export async function createVehicle(payload: Partial<Vehicle>): Promise<Vehicle> {
  const { data } = await api.post<Vehicle>("/vehicles", payload);
  return data;
}

export async function updateVehicle(id: string, patch: Partial<Vehicle>): Promise<Vehicle> {
  const { data } = await api.put<Vehicle>(`/vehicles/${id}`, patch);
  return data;
}

export async function deleteVehicle(id: string): Promise<void> {
  await api.delete(`/vehicles/${id}`);
}