const isoNow = () => new Date().toISOString();

const asJson = (value) => JSON.stringify(value, null, 2);
const fromJson = (value) => JSON.parse(value);
const SUBSCRIPTION_STATUS_PRIORITY = {
  incomplete: 1,
  incomplete_expired: 0,
  trialing: 4,
  active: 5,
  past_due: 2,
  canceled: 1,
  unpaid: 1,
  paused: 1,
  checkout_completed: 3,
};

const normalizePeriodEnd = (value) => {
  if (value == null || value === "") {
    return null;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return null;
  }
  return new Date(numeric * 1000).toISOString();
};

const firstPriceId = (subscription) => {
  const items = subscription?.items?.data;
  if (!Array.isArray(items) || !items.length) {
    return null;
  }
  return items[0]?.price?.id || null;
};

export const getSubscriberStateStore = (env) => {
  const store = env.SUBSCRIBER_STATE;
  if (!store || typeof store.put !== "function" || typeof store.get !== "function") {
    return null;
  }
  return store;
};

export const buildSubscriberRecord = (event) => {
  const object = event?.data?.object || {};
  const updatedAt = isoNow();

  if (event?.type === "checkout.session.completed") {
    return {
      customer_id: object.customer || null,
      subscription_id: typeof object.subscription === "string" ? object.subscription : null,
      status: "checkout_completed",
      price_id: null,
      current_period_end: null,
      updated_at: updatedAt,
      checkout_session_id: object.id || null,
      email: object.customer_details?.email || object.customer_email || null,
      source_event_id: event.id || null,
      source_event_type: event.type,
      source_event_created: normalizePeriodEnd(event.created),
    };
  }

  if (event?.type?.startsWith("customer.subscription.")) {
    return {
      customer_id: object.customer || null,
      subscription_id: object.id || null,
      status: object.status || null,
      price_id: firstPriceId(object),
      current_period_end: normalizePeriodEnd(object.current_period_end),
      updated_at: updatedAt,
      checkout_session_id: null,
      email: null,
      source_event_id: event.id || null,
      source_event_type: event.type,
      source_event_created: normalizePeriodEnd(event.created),
    };
  }

  return null;
};

export const persistSubscriberRecord = async (store, record) => {
  if (!store) {
    throw new Error("SUBSCRIBER_STATE binding is unavailable.");
  }
  if (!record?.customer_id) {
    throw new Error("Subscriber record is missing customer_id.");
  }
  if (!record?.subscription_id && record?.status !== "checkout_completed") {
    throw new Error("Subscriber record is missing subscription_id.");
  }

  let recordToPersist = record;

  if (record.subscription_id) {
    const subscriptionKey = `subscription:${record.subscription_id}`;
    const existingRaw = await store.get(subscriptionKey);
    if (existingRaw) {
      const existing = fromJson(existingRaw);
      const existingPriority = SUBSCRIPTION_STATUS_PRIORITY[existing?.status] ?? -1;
      const incomingPriority = SUBSCRIPTION_STATUS_PRIORITY[record?.status] ?? -1;

      // Never let a weaker checkout-completed snapshot overwrite a stronger
      // subscription-state record such as active or trialing.
      if (existingPriority > incomingPriority) {
        recordToPersist = {
          ...existing,
          customer_id: existing.customer_id || record.customer_id,
          subscription_id: existing.subscription_id || record.subscription_id,
          checkout_session_id: record.checkout_session_id || existing.checkout_session_id || null,
          email: existing.email || record.email || null,
          updated_at: record.updated_at || existing.updated_at,
          source_event_id: record.source_event_id || existing.source_event_id || null,
          source_event_type: record.source_event_type || existing.source_event_type || null,
          source_event_created: record.source_event_created || existing.source_event_created || null,
        };
      }
    }

    await store.put(subscriptionKey, asJson(recordToPersist));
  }

  await store.put(`customer:${record.customer_id}`, asJson(recordToPersist));

  return {
    customer_key: `customer:${record.customer_id}`,
    subscription_key: record.subscription_id ? `subscription:${record.subscription_id}` : null,
  };
};

export const loadSubscriberRecordBySubscriptionId = async (store, subscriptionId) => {
  if (!store) {
    throw new Error("SUBSCRIBER_STATE binding is unavailable.");
  }
  if (!subscriptionId) {
    return null;
  }

  const raw = await store.get(`subscription:${subscriptionId}`);
  if (!raw) {
    return null;
  }

  return fromJson(raw);
};
