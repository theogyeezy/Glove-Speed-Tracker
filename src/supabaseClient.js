import { createClient } from '@supabase/supabase-js';

// Initialize Supabase client
const supabaseUrl = 'https://hcahfchcgzntcylbnokb.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhjYWhmY2hjZ3pudGN5bGJub2tiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM3MjQyMzMsImV4cCI6MjA1OTMwMDIzM30.MaORy3NSEkFEJtSfxFd9MuIvF6NJwHp4SHA40ydE7og';

// Create and export the Supabase client as a named export
export const supabase = createClient(supabaseUrl, supabaseKey);

// Also export as default for backward compatibility
export default supabase;
